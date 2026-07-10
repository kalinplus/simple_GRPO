from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import json, os, shutil, re, random, io, requests, ctypes, sys, time, struct, subprocess
import swanlab
import torch
import torch.nn as nn
import numpy as np
import torch.distributed as dist
import torch.multiprocessing as mp
from tqdm import tqdm
os.environ['TOKENIZERS_PARALLELISM'] = 'true'

model_path = "./models/Qwen/Qwen2.5-3B"
gen_device = 7    # GPU device for generation, don't put it in CUDA_VISIBLE_DEVICES
beta = 0.04
all_steps = 1000
Q_batch_size = 5
num_pre_Q = 8
train_batch_size = 1
gen_update_steps = 16  # gen model 每 16 步同步一次参数，在线和离线生成数据的中间状态。注意和 grad acc = 16 是相同的，也就是每真正更新一次参数，就同步 gen model
save_steps = 200
compute_gen_logps = True
clip_param = 0.2
ref_server = "http://localhost:59875"

# ---- Reward 配置（对应 docs/GRPO_LEARNING_GUIDE.md §6.4 reward 设计实验）----
# 结果（正确性）reward 的开关与数值
use_correct_reward = True
reward_correct_right = 1.0   # 答对给多少
reward_correct_wrong = -1.0  # 答错 / 没提取到数字给多少
# 格式 reward 的开关与数值
use_format_reward = True
reward_format_right = 1.25   # <think>...</think><answer>...</answer> 格式正确给多少
reward_format_wrong = -1.0   # 格式不对给多少

# ---- Checkpoint 保存配置 ----
# ckpt 存到 {save_dir}/{run_tag}/step_{step}。run_tag=None 时用 build_run_tag() 自动拼接常用参数，
# 这样不同实验（beta / num_pre_Q / lr / reward 配置）互不覆盖；想自定义直接给 run_tag 赋字符串即可。
save_dir = "./ckpts"
run_tag = None

from ref_server import tensor_to_bytes, bytes_to_tensor, make_bytes_list, bytes_list_to_list

def log_gpu_memory():
    r = subprocess.run(['nvidia-smi', '--query-gpu=index,memory.used,memory.total', '--format=csv,noheader,nounits'], capture_output=True, text=True)
    for line in r.stdout.strip().split('\n'):
        gpu_id, used, total = line.split(',')
        print(f"  GPU {gpu_id.strip()}: {used.strip()} / {total.strip()} MiB")

ds_config = {
    "train_micro_batch_size_per_gpu": train_batch_size,
    "gradient_accumulation_steps": 16,  # 这样做 1000 steps, 其实等价于 batch=16 做 62 个 steps，消耗数据量是相同的
    "optimizer": {
        "type": "AdamW",
        "params": { "lr": 1e-6 }
    },
    "bf16": {"enabled": True},
    "zero_optimization": {
        "stage": 2,
        "allgather_partitions": True,
        "allgather_bucket_size": 2e8,
        "overlap_comm": True,
        "reduce_scatter": True,
        "reduce_bucket_size": 2e8,
        "contiguous_gradients": True,
        "stage3_gather_16bit_weights_on_model_save": True,
        "offload_optimizer": {"device": "cpu"}
    }
}

def build_run_tag():
    """根据常用参数自动拼一个 run_tag，用作 ckpt 子目录名和 swanlab 实验名后缀，
    让不同实验（beta / num_pre_Q / lr / reward 配置）存到不同目录、互不覆盖。"""
    model_name = os.path.basename(model_path.rstrip('/'))
    lr = ds_config["optimizer"]["params"]["lr"]
    parts = [model_name, f"beta{beta}", f"G{num_pre_Q}", f"lr{lr:g}"]
    if not use_correct_reward and not use_format_reward:
        parts.append("noreward")
    else:
        rsig = []
        if use_correct_reward: rsig.append(f"cor{reward_correct_right:g}")
        if use_format_reward:  rsig.append(f"fmt{reward_format_right:g}")
        parts.append("+".join(rsig))
    return "_".join(parts)

def get_batch():
    """
    从 refer server /get 得到训练数据
    """
    try:
        r = requests.get(f"{ref_server}/get").content
        if r == b'empty': return None
    except: return None
    dd = bytes_list_to_list(r)
    data = json.loads(dd[0]) 
    data['inputs'] = bytes_to_tensor(dd[1])
    data['rewards'] = bytes_to_tensor(dd[2])
    data['refs'] = bytes_to_tensor(dd[3])
    if len(dd) == 5: data['gen_logps'] = bytes_to_tensor(dd[4])
    return data

def get_per_token_logps(logits, input_ids):
    """
    gather the log probabilities for each token in input_ids (correct answers) from the logits
    @param logits: (B, L-1, V), the logits for each token
    @param input_ids: (B, L-1), the input token ids for which to gather log probabilities
    @return: (B, L-1), the log probabilities for each token in input_ids
    """
    per_token_logps = [] # Use a loop to reduce memory peak.
    for logits_row, input_ids_row in zip(logits, input_ids):  # for each sample in the batch
        log_probs = logits_row.log_softmax(dim=-1)  # logits 是未经 softmax 的
        token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)  # (L-1,)
        per_token_logps.append(token_log_prob)
    return torch.stack(per_token_logps)  # (B, L-1)
#from kernel.ce_kernel import fast_log_softmax_gather
#get_per_token_logps = fast_log_softmax_gather

def GRPO_step(batch):
    """
    calculate GRPO loss

    @param: batch(dict): contains batch of prompts lengths, input ids, normalized advantanges, ref logps
    """
    prompt_length = batch['plen']  # (B,), prompt length
    inputs = batch['inputs'].to(engine.device)  # (B, L): prompt+answer 的 token ids
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)  # (B, 1), normalized advantages，是每条回答一个
    logits = engine(inputs).logits
    logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
    input_ids = inputs[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it 
    per_token_logps = get_per_token_logps(logits, input_ids)  # (B, L-1), 每个位置的 log probability
    per_token_logps = per_token_logps[:,prompt_length-1:]  # (B, answer_length), 只保留 answer 部分的 log probability, 由 \pi_\theta 生成
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)  # (B, answer_length), 由 reference model 生成的 log probability
    # KL 散度的展开近似: KL(p || q) = \sum p * (log p - log q) ≈ exp(log q - log p) - (log q - log p) - 1，是逐 token 的
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()  # (B, L), 掩码，只计算 answer 部分的 loss，忽略 padding 和 prompt 部分
    if 'gen_logps' in batch:
        # batch['gen_logps'] 是由 gen worker 中的模型产生的，每 gen_update_steps, 训练 proc 会把模型参数发给 gen worker，gen worker 用其（介于 actor 和 ref 之间的半旧模型）生成样本并计算 log probability 作为 \pi_{old} 的输出
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))  # (B, answer_length), radio = \pi_\theta / \pi_{old}
        # 以下两步和 PPO 的 clip 一样
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)  # (B, answer_length), advantages 被广播到了每个 token 的 loss
    else: 
        # fallback：没有 gen_logps 时，ratio 恒为 1
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        assert compute_gen_logps is False
    per_token_loss = -(per_token_loss - beta * per_token_kl)  # 每个 token 的 loss
    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()  # (1,) 每条回答的 loss，对 answer 长度做了归一化，然后对 batch 做平均
    avg_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()  # (1,) 平均每个 token 的 KL 散度
    return loss, avg_kl


def gen_worker(Q, physics_device, info_Q):
    """
    生成 worker：采样 prompts, 生成 num_pre_Q 个回答，计算奖励（和可能的 gen logps(用于计算 ppo-style 的 importance radio))，上传到 ref_server
    """
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["CUDA_VISIBLE_DEVICES"] = f'{physics_device}'
    cleanup_keys = [  
            'RANK', 'WORLD_SIZE', 'MASTER_ADDR', 'MASTER_PORT', 'LOCAL_RANK',  
            'LOCAL_WORLD_SIZE', 'GROUP_RANK', 'ROLE_RANK', 'ROLE_NAME',   
            'GROUP_WORLD_SIZE', 'ROLE_WORLD_SIZE',  
            'TORCHELASTIC_RESTART_COUNT', 'TORCHELASTIC_MAX_RESTARTS',  
            'TORCHELASTIC_RUN_ID', 'TORCHELASTIC_USE_AGENT_STORE',  
            'TORCHELASTIC_ERROR_FILE',  
            'TORCH_NCCL_ASYNC_ERROR_HANDLING',  
            'NCCL_COMM_ID', 'NCCL_DEBUG', 'NCCL_SOCKET_IFNAME',  
        ]  
    for key in cleanup_keys: os.environ.pop(key, None)
    torch.cuda.set_device(0)
    print(f"Generation worker process uses GPU {physics_device}")
    
    from vllm import LLM, SamplingParams
    vllm_gen = LLM(model=model_path, gpu_memory_utilization=0.4)
    ref_server_ver = 'tensor'  # don't worry, it will auto switch based on the first upload

    # ---- 在采样阶段就禁止生成“死区” token，从源头杜绝越界 id ----
    # Qwen2.5 的 lm_head/embedding 被填充到 64 的倍数（config.vocab_size=151936），
    # 而 tokenizer 实际定义的 token 数（len(tokenizer)）更小，二者之间的 id 区间
    # 没有任何字符串含义——是“死区”。但它们仍是合法的 embedding 行，所以采样器可能
    # 把它们挑出来；一旦这些 id 回灌到第二次 vLLM pass（作为 prompt_token_ids 计算
    # prompt_logprobs）就会触发 V1 的 `max_input_id > tokenizer.max_token_id` 校验而
    # 崩溃（“Token id X is out of vocabulary”）。
    #
    # 用 logit_bias 把死区 id 钳到 -100（softmax 后概率≈0，等价于硬掩码）。这是 vLLM
    # V1 原生支持的写法——V1 拒绝 per-request 的 logits_processors，但接受 logit_bias
    # （仅校验 id 落在 [0, vocab_size) 内，由引擎内部用 GPU 算子批量施加）。
    full_vocab_size = AutoConfig.from_pretrained(model_path).vocab_size   # 填充后的词表大小
    valid_vocab_size = len(tokenizer)                                     # tokenizer 真正定义的 token 数
    if valid_vocab_size < full_vocab_size:
        dead_zone_bias = {tid: -100.0 for tid in range(valid_vocab_size, full_vocab_size)}
        print(f'[VLLM PROC] logit_bias forbids dead-zone ids [{valid_vocab_size}, {full_vocab_size}) '
              f'({len(dead_zone_bias)} ids)')
    else:
        dead_zone_bias = None  # 词表无填充死区时，不施加任何 bias

    sampling_params = SamplingParams(n=num_pre_Q, temperature=0.9, max_tokens=700, logit_bias=dead_zone_bias)
    gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)

    from datasets import load_dataset
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    QAs = [{'Q':x, 'A':y.split('####')[-1].strip()} for x,y in zip(dataset['question'], dataset['answer'])]
    
    system_prompt = """You are a helpful assistant. A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The Assistant first thinks about the reasoning process in the mind and then provides the user with the answer.\
    The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>."""

    def gen_answers(prompts):
        """
        输入 prompts，将其应用对话模板，tokenize 后送入 vLLM 生成模型，返回生成的 answers 和对应的 token ids
        """
        tip_text = []
        for x in prompts:
            tip_text.append(tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True))
        voutputs = vllm_gen.generate(tip_text, sampling_params, use_tqdm=False)
        answers = [];  ans_token_ids = []
        for v in voutputs:
            for z in v.outputs: 
                answers.append(z.text)
                ans_token_ids.append(z.token_ids)
        return answers, ans_token_ids

    from math_verify import parse, verify, ExprExtractionConfig
    # parse 把字符串解析为 SymPy 数学表达式对象，这样就能做数学等价性比较，不只是字符串匹配（比如 0.5 = 1/2 = 2^{-1})
    # SymPy 的核心特点是可以定义符号变量，进行符号运算和简化，支持代数、微积分、方程求解等数学操作
    def reward_correct(item, answer):
        pattern = r'\d+\.\d+|\d+/\d+|\d+'
        nums = re.findall(pattern, answer)
        if len(nums) == 0: return reward_correct_wrong
        lastnum = nums[-1]
        ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
        ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
        return reward_correct_right if verify(ans, ground_truth) else reward_correct_wrong

    def reward_format(item, answer):
        pattern = r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$"  # 严格匹配 <think>...</think> <answer>...</answer>，中间可以有换行和空格
        think_count = answer.count("<think>") + answer.count("</think>")
        answer_count = answer.count("<answer>") + answer.count("</answer>")
        return reward_format_right if re.match(pattern, answer, re.DOTALL | re.VERBOSE) and think_count==2 and answer_count==2 else reward_format_wrong


    def gen_samples(inputs):
        """
        输入 prompts (inputs), 按 num_pre_Q 作为 batch 生成答案，计算奖励，返回 prompts_text, rewards, answers, ans_token_ids
        """
        prompts = [x["Q"] for x in inputs]
        answers, ans_token_ids = gen_answers(prompts)
        rewards = []
        for i, inp in enumerate(inputs):
            for a in answers[i*num_pre_Q:(i+1)*num_pre_Q]:
                # 根据答案，按开关用结果/格式验证器计算奖励
                r = 0.0
                if use_correct_reward: r += reward_correct(inp, a)
                if use_format_reward:  r += reward_format(inp, a)
                rewards.append(r)
        prompts_text = [tokenizer.apply_chat_template([
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": x}], tokenize=False, add_generation_prompt=True) for x in prompts]
        return prompts_text, torch.tensor(rewards, dtype=torch.float32), answers, ans_token_ids

    def try_update_model():
        try:
            new_state_dict = Q.get_nowait()
            print('[VLLM PROC] recving new model ...')
            llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights(new_state_dict.items())
            print('[VLLM PROC] model updated')
            del new_state_dict
        except:
            #print('[VLLM PROC] no new model')
            return
        
    from torch.nn.utils.rnn import pad_sequence
    for it in range(999999999):
        if it % 3 == 0: try_update_model()
        inputs = random.sample(QAs, Q_batch_size)
        tic = time.time()
        prompt_inputs, rewards, answers, ans_token_ids = gen_samples(inputs)
        print(f'time: {time.time()-tic:.2f}s    ', 'rewards:', rewards, )
        if it % 5 == 0: print('answers:', answers[0])
        rw = torch.tensor(rewards)
        info_Q.put({'reward_mean': rw.mean().item(), 'reward_max': rw.max().item(),
                     'reward_min': rw.min().item(), 'reward_acc': (rw > 0).float().mean().item(),
                     'sample_answer': answers[0][:500]})

        for i, pp in enumerate(prompt_inputs):
            prompt_ids = tokenizer(pp, return_tensors="pt", add_special_tokens=False)["input_ids"]
            plen = prompt_ids.shape[1]  # prompt length
            # 每个 prompt 生成 num_pre_Q 个答案，计算奖励后，按 prompt 分组上传到 ref_server
            # 所有 prompt 的 num_pre_Q 个回答都展平放在一个列表里，所以要按 num_pre_Q 分片
            # 恢复 prmopt -> num_pre_Q 个回答的对应关系
            curr_answers = answers[i*num_pre_Q:(i+1)*num_pre_Q]
            curr_ans_ids = ans_token_ids[i*num_pre_Q:(i+1)*num_pre_Q]
            curr_rewards = rewards[i*num_pre_Q:(i+1)*num_pre_Q]
            if curr_rewards.max() - curr_rewards.min() < 1e-4: continue

            if ref_server_ver == 'tensor':  # 传给 ref server, 直接传 token ids tensor 是正常高效路径
                # 计算 advantanges
                curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)
                # 把生成好的样本按 train_batch_size 分片上传到 ref_server
                for ii in range(0, num_pre_Q, train_batch_size):
                    sub_rewards = curr_rewards[ii:ii+train_batch_size]
                    sub_ans_ids = curr_ans_ids[ii:ii+train_batch_size]
                    tensor_list = [torch.tensor(lst) for lst in sub_ans_ids]
                    output_ids = pad_sequence(tensor_list, batch_first=True, padding_value=tokenizer.pad_token_id)  # batch 内部 padding 对齐
                    Qrep = prompt_ids.repeat(1, output_ids.shape[0]).view(-1, plen)  # prompts 复制 B 份对齐
                    merged_ids = torch.cat([Qrep, output_ids], dim=1)  # 拼接为完整 prompt + answer 的 token ids
                    data = [json.dumps({"plen": plen}).encode(), tensor_to_bytes(merged_ids), tensor_to_bytes(sub_rewards)]       

                    # 如果要计算 generator 的 log-probs, 就用 vllm 重新算一遍
                    # 用在训练时，计算重要性采样的 radio
                    if compute_gen_logps:
                        zz = vllm_gen.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp, use_tqdm=False)
                        zz = [xx.prompt_logprobs[plen:] for xx in zz]
                        gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
                        data.append(tensor_to_bytes(gen_logps))

                    xdata = make_bytes_list(data)
                    r = requests.post(f"{ref_server}/upload", data=xdata)
                    if r.content == b'string': ref_server_ver = 'string'
            elif ref_server_ver == 'string':
                xdata = make_bytes_list([json.dumps({"Q": pp[0], "As": curr_answers}).encode(), 
                                        tensor_to_bytes(curr_rewards)])
                r = requests.post(f"{ref_server}/upload", data=xdata)
                if r.content == b'tensor': ref_server_ver = 'tensor'


tokenizer = AutoTokenizer.from_pretrained(model_path)
if __name__ == '__main__':
    import deepspeed
    deepspeed.init_distributed()

    if dist.get_rank() == 0:
        # ref 和 gen worker 都只在 rank0, 所以同步 gen worker 也只能在 rank0
        print('\nSTART vLLM generation...\n')
        mp.set_start_method('spawn')
        Q = mp.Queue()
        info_Q = mp.Queue()
        def launch_gen():
            pp = mp.Process(target=gen_worker, args=(Q, gen_device, info_Q))
            pp.start()
            return pp
        p = launch_gen()  # 启动 gen worker 进程，生成样本并上传到 ref_server
        if run_tag is None: run_tag = build_run_tag()
        print(f'[RUN] run_tag = {run_tag}    (ckpt -> {save_dir}/{run_tag}/step_*)')
        swanlab.init(project="GRPO-GSM8K", experiment_name=f"{run_tag}_{time.strftime('%m%d_%H%M')}",
                      config={'model': model_path, 'run_tag': run_tag,
                              'lr': ds_config['optimizer']['params']['lr'], 'beta': beta, 'clip': clip_param,
                              'batch_size': train_batch_size, 'grad_acc': ds_config['gradient_accumulation_steps'],
                              'gen_update_steps': gen_update_steps, 'num_pre_Q': num_pre_Q,
                              'Q_batch_size': Q_batch_size, 'max_tokens': 700,
                              'use_correct_reward': use_correct_reward, 'reward_correct_right': reward_correct_right,
                              'reward_correct_wrong': reward_correct_wrong,
                              'use_format_reward': use_format_reward, 'reward_format_right': reward_format_right,
                              'reward_format_wrong': reward_format_wrong})

    model = AutoModelForCausalLM.from_pretrained(model_path, 
            torch_dtype=torch.bfloat16, _attn_implementation="sdpa")

    # engine 是 deepspeed 对 model (AutoModelForCausalLM) 的封装，optimizer 是 deepspeed 对 optimizer 的封装
    engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, 
                                                model_parameters=model.parameters())

    progress = range(1, all_steps+1)
    if dist.get_rank() == 0: progress = tqdm(progress)  # dist.get_rank() == 0 的进程才显示进度条，避免多进程同时打印干扰
    for step in progress:
        batch = get_batch()
        while batch is None:
            print('waiting for batch...'); time.sleep(1)
            batch = get_batch()

        loss, avg_kl = GRPO_step(batch)  # 计算 GRPO loss 和平均 KL 散度
        engine.backward(loss)  # 算出 grad
        engine.step()  #  更新参数

        if dist.get_rank() == 0:
            progress.set_description(f"Loss: {loss.item():.6f}")
            # collect latest gen info
            gen_info = {}
            while True:
                try: gen_info = info_Q.get_nowait()
                except: break
            log_data = {'train/loss': loss.item(), 'train/kl': avg_kl.item()}
            log_data.update({f'reward/{k}': v for k, v in gen_info.items() if k != 'sample_answer'})
            # 采样一条回答看看情况
            if step % 50 == 0 and 'sample_answer' in gen_info:
                log_data['sample/answer'] = swanlab.Text(gen_info['sample_answer'])
            # 看看 gpu 占用
            if step % 10 == 0:
                log_gpu_memory()
                r = subprocess.run(['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
                                     capture_output=True, text=True)
                mems = [float(x.strip()) for x in r.stdout.strip().split('\n') if x.strip()]
                log_data['gpu/max_memory_mb'] = max(mems) if mems else 0
            swanlab.log(log_data, step=step)
            if not p.is_alive():
                print(f'[WARNING] gen_worker died at step {step}, restarting...')
                p = launch_gen()

        if step % gen_update_steps == 0:
            # 同步 gen worker 参数
            dist.barrier()  # 同步栅栏，强制同步所有分片的状态
            if dist.get_rank() == 0:
                print('[TRAINING PROC] sending latest state_dict ...')
                state_dict = engine.module.state_dict()
                Q.put(state_dict)
                print('[TRAINING PROC] send state_dict ok!')
            dist.barrier()  # 所以 rank0 还在往 Q 里放 state_dict, 其他 rank 等待，确保其他 rank 不会提前 all-reduce 和等待，两边都要求写，死锁

        if step % save_steps == 0:
            # 保存 ckpt 到 {save_dir}/{run_tag}/step_{step}，run_tag 由常用参数拼接，避免不同实验互相覆盖
            dist.barrier()
            if dist.get_rank() == 0:
                save_name = f"{save_dir}/{run_tag}/step_{step}"
                print(f'saving model to {save_name}')
                state_dict = engine.module.state_dict()
                state_dict = type(state_dict)({k: v.cpu() for k, v in state_dict.items()})
                engine.module.save_pretrained(save_name, state_dict=state_dict)
                tokenizer.save_pretrained(save_name)
            dist.barrier()

    if dist.get_rank() == 0:
        swanlab.finish()
