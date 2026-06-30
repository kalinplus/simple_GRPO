# simple_GRPO — GRPO 学习指南（从看懂到实操）

> 面向：想把这个项目当作 GRPO 学习载体、并准备自己从头训练一遍的读者。
> 定位：本文是 `CLAUDE.md` 的「学习向」展开版（`CLAUDE.md` 是「开发向」速查）。
> 阅读建议：先看第 1、3 节建立算法认知 → 第 2 节对照代码 → 第 4 节准备环境 → 第 5 节动手跑 → 第 6 节调参实验 → 第 7 节看变体对照。

---

## 0. 一句话定位

本项目是一个**极简的 GRPO（Group Relative Policy Optimization）实现**，约 200 行核心代码，无 ray/accelerate/trl 依赖——只有 `deepspeed`、`torch`、`vllm`、`transformers`、`bottle`。默认任务是 GSM8K 数学推理，reward = 数学正确性 + 回答格式合规。它是理解 R1 风格「`.<answer>` 推理链」训练过程的最佳教材之一。

核心设计：**三进程架构**（训练进程 + 生成 Worker + 参考模型服务器），通过 HTTP + multiprocessing Queue 通信，解耦了生成、参考对数概率计算和训练。

---

## 1. 先理清概念：GRPO 到底在做什么？

### 1.1 从 RLHF 到 GRPO 的思维链条

**传统 RLHF**：
1. SFT：让基座模型学会对话格式
2. Reward Model（RM）：训练一个打分模型
3. PPO：用 RM 的分数作为 reward，做在线策略优化

**PPO 的痛点**：需要一个独立的 Reward Model（大、贵、不稳定），在线采样开销大。

**GRPO 的突破**（DeepSeek-R1 论文）：
- **不需要 Reward Model**，直接用规则/验证器（rule-based verifier）给 reward。比如数学题，直接算答案对不对。
- **不需要 Critic Model**（value function），把**同一道题的多个回答**组成一个 group，reward 在 group 内做相对比较，得到 advantage。
- 所以 GRPO = **去掉 RM + 去掉 Critic 的极简 PPO**，只需要一个 policy model 和一个 ref model。

### 1.2 GRPO 的数学公式

给定 prompt $x$，生成 $G$ 个回答 $\{y_1, ..., y_G\}$，每个回答获得 reward $r_i$。

**Step 1 — 计算 advantage（组内归一化）**：
$$A_i = \frac{r_i - \text{mean}(\{r_j\})}{\text{std}(\{r_j\}) + \epsilon}$$

就是把 reward 减去组均值、除以组标准差。这样正确回答 advantage 为正，错误回答 advantage 为负。

**Step 2 — 计算重要性比率（importance ratio）**：
$$\rho_t = \frac{\pi_\theta(y_t | y_{<t}, x)}{\pi_{\text{old}}(y_t | y_{<t}, x)}$$

其中 $\pi_{\text{old}}$ 是**生成时**的旧策略（不是初始 ref 策略）。这个比率衡量「当前策略 vs 生成时的策略」偏离了多少。

**Step 3 — PPO-style clipped objective**：
$$L_i = \min\left(\rho_t \cdot A_i,\; \text{clip}(\rho_t, 1-\epsilon, 1+\epsilon) \cdot A_i\right)$$

clip 防止策略更新太激进（和 PPO 一样的思想）。

**Step 4 — 加 KL 惩罚**：
$$L = -\left(L_i - \beta \cdot D_{\text{KL}}(\pi_\theta \| \pi_{\text{ref}})\right)$$

$D_{\text{KL}}$ 是当前策略和**初始冻结策略** $\pi_{\text{ref}}$ 之间的 KL 散度，防止模型偏离太远。

**最终 loss**：对 completion（回答）部分的 token 取平均，再对 batch 内的样本取平均。

### 1.3 GRPO vs PPO vs REINFORCE++ 对比

| 组件 | PPO | GRPO | REINFORCE++ |
|---|---|---|---|
| Reward Model | ✅ 单独训练 | ❌ 规则/验证器 | ❌ 规则/验证器 |
| Critic (Value) | ✅ | ❌ 组内归一化 | ❌ |
| Advantage 来源 | GAE (λ-return) | 组内 reward 归一化 | 组内 reward 归一化 |
| Reference Model | ✅ | ✅ (KL 惩罚) | ❌ (用 gen_logps 替代) |
| 重要性比率 | $\pi_\theta / \pi_{\text{old}}$ | $\pi_\theta / \pi_{\text{old}}$ | $\pi_\theta / \pi_{\text{old}}$ |
| Clip | ✅ | ✅ | ✅ |
| 核心思想 | 在线 RL | group 内做 competition | 去掉 ref，更简洁 |

### 1.4 为什么 GRPO 能教会模型「思考」？

关键在于 **advantage 信号的设计**：
- 正确答案 → advantage > 0 → 增大这些 token 的概率
- 错误答案 → advantage < 0 → 降低这些 token 的概率
- 格式正确的 → 额外奖励 → 强化 `.<answer>` 输出格式

经过大量训练后，模型逐渐学会：「先在 ` 里一步一步推理，再在 `<answer>` 里给出最终答案」这种思维链模式能获得更高的 reward。

---

## 2. 代码阅读地图（按推荐顺序）

| 顺序 | 文件/位置 | 作用 | 学 GRPO 时重点看 |
|---|---|---|---|
| ① | `grpo_vllm_one.py:76-98` | **GRPO loss**，全文核心 | `GRPO_step()`：advantage、ratio、clip、KL，20 行搞定 |
| ② | `grpo_vllm_one.py:66-72` | `get_per_token_logps` | per-token log 概率的计算方式 |
| ③ | `grpo_vllm_one.py:144-157` | Reward 函数 | `reward_correct` + `reward_format`，理解 reward 怎么来的 |
| ④ | `grpo_vllm_one.py:160-229` | `gen_worker`：生成 Worker | vLLM 生成 → 计算 reward → 组装 batch → POST 给 ref_server |
| ⑤ | `ref_server.py:59-91` | HTTP 服务：接收 batch + 算 ref logps | `/upload` 接收 → 算 ref logps → 放 result_queue → `/get` 取出 |
| ⑥ | `grpo_vllm_one.py:53-64` | `get_batch` | 训练进程从 ref_server GET 带有 ref logps 的完整 batch |
| ⑦ | `grpo_vllm_one.py:12-24` | 硬编码参数 | `beta`、`clip_param`、`gen_update_steps` 等核心超参 |
| ⑧ | `grpo_vllm_one.py:232-315` | **主训练循环** | `deepspeed` 初始化、三进程启动、训练+保存+同步 |
| ⑨ | `ref_server.py:5-25` | 序列化工具 | `tensor_to_bytes` / `bytes_to_tensor` / `make_bytes_list`，进程间数据传输 |
| ⑩ | `simple-reinforce++/rf++_vllm_one.py:52-72` | **REINFORCE++ loss** | 与 GRPO 的对比：无 KL、per-token advantage、不同归一化 |

> 顺序 ①→③→④→⑤ 一口气读完，GRPO 的完整数据流就吃透了；②⑥⑦是工程细节。

---

## 3. 架构与算法逐行拆解

### 3.1 三进程架构——理解数据流（最重要的一节）

```
┌─────────────────┐    HTTP POST /upload    ┌──────────────────┐
│   gen_worker     │  ──────────────────►   │   ref_server     │
│  (vLLM, GPU G)   │  batch+rewards+gen_logps│  (frozen model,  │
│                  │  ◄──────────────────   │   GPU R)         │
│  - vLLM 生成回答  │   return b'tensor'     │                  │
│  - 计算 reward   │                         │  - 算 ref logps  │
│  - 算 gen_logps  │                         │  - 放 result_queue│
│  - 组装 tensor   │                         │                  │
└────────┬─────────┘                         └────────┬─────────┘
         │                                              │
         │ mp.Queue (state_dict)                        │ HTTP GET /get
         │ (每 gen_update_steps 同步一次)               │
         ▼                                              ▼
┌─────────────────┐                         ┌──────────────────┐
│  Training Proc   │ ◄── 完整 batch ────────│  result_queue    │
│  (DeepSpeed,     │                         │  (内存队列)       │
│   GPU 0..N)      │                         └──────────────────┘
│                  │
│  - GRPO_step()  │
│  - backward      │
│  - 定期推 state_dict 到 Queue
└─────────────────┘
```

**数据流详解**：

1. **gen_worker** 从 GSM8K 随机抽题 → vLLM 每题生成 `num_pre_Q`（默认 8）个回答
2. 对每个回答计算 reward（正确性 + 格式），跳过 reward 方差太小的组（没有优势信号）
3. 组内归一化 reward（减均值除标准差），用 vLLM 的 `prompt_logprobs` 算生成时的 log 概率
4. 把 `[prompt+answer token_ids, rewards, gen_logps]` 序列化为 bytes → **HTTP POST** 给 ref_server
5. **ref_server** 收到后，用冻结模型算每个 token 的 ref log 概率，拼上之前的字段，放入 `result_queue`
6. **训练进程** 通过 **HTTP GET** 从 result_queue 拿到完整 batch（含 ref logps），跑 `GRPO_step` → backward → step
7. 每 `gen_update_steps` 步，rank 0 把最新 `state_dict` 放入 `mp.Queue` → gen_worker 取出 → vLLM 热更新权重

**为什么这样设计？**
- **ref_server 独立进程**：避免 DeepSpeed 多卡复制 ref model 到每个 rank，省显存；可以跑在另一台机器上
- **gen_worker 独立 GPU**：vLLM 需要自己的显存管理，不能和 DeepSpeed 训练进程共享 GPU
- **HTTP 通信**：最简单的跨进程协议，比 NCCL 简单得多，调试方便
- **mp.Queue 同步权重**：训练进程只需要每 N 步同步一次 state_dict，开销不大

### 3.2 GRPO loss 逐行拆解 —— `grpo_vllm_one.py:76-98`

```python
def GRPO_step(batch):
    prompt_length = batch['plen']
    inputs = batch['inputs'].to(engine.device)       # (B, L): prompt+answer 的 token ids
    advantages = batch['rewards'].to(engine.device).unsqueeze(1)  # (B, 1): 已归一化的 advantage
    logits = engine(inputs).logits                    # (B, L, V): 当前策略的 logits
    logits = logits[:, :-1, :]                        # 去掉最后一个位置（没有下一个 token）
    input_ids = inputs[:, 1:]                         # 去掉第一个 token（没有对应的 logits）
    per_token_logps = get_per_token_logps(logits, input_ids)  # (B, L-1): 每个位置的 log p
    per_token_logps = per_token_logps[:, prompt_length-1:]     # 只保留 answer 部分的 logps
```

到这里，`per_token_logps` 就是一个 (B, answer_length) 的矩阵，每个元素是当前策略 $\pi_\theta$ 在对应位置生成该 token 的 log 概率。

```python
    ref_per_token_logps = batch['refs'].to(per_token_logps.device)  # ref 策略的 logps
    # KL 散度的展开形式: KL(p || q) = Σ p*(log p - log q) ≈ exp(log q - log p) - (log q - log p) - 1
    # 这里: KL(π_ref || π_θ) 用的是展开近似
    per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) \
                   - (ref_per_token_logps - per_token_logps) - 1
```

**KL 散度的近似计算**。标准的 KL 散度是 $D_{\text{KL}}(p \| q) = \sum p \log(p/q)$，但逐 token 精确算需要 softmax 出完整的概率分布，非常耗显存。这里用了一阶 Taylor 展开：$e^x - x - 1 \approx x^2/2$，当 $x$（即 ref_logps - logps）接近 0 时近似成立。

```python
    completion_mask = (inputs[:, prompt_length:] != tokenizer.pad_token_id).int()
```

掩码：只计算 answer 部分（非 padding）的 loss，prompt 部分不参与优化。

```python
    if 'gen_logps' in batch:
        ratio = torch.exp(per_token_logps - batch['gen_logps'].to(engine.device))
        # ratio = π_θ / π_old，衡量当前策略 vs 生成时的策略偏离了多少
        clipped_ratio = torch.clamp(ratio, 1-clip_param, 1+clip_param)
        # clip 到 [0.8, 1.2]，防止一步更新太大
        per_token_loss = torch.min(ratio * advantages, clipped_ratio * advantages)
    else:
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages
        # fallback：没有 gen_logps 时，ratio 恒为 1（因为 detach 后等于自己）
        # 此时退化为普通 REINFORCE：per_token_loss ≈ advantages
```

**这是 PPO 的 clipped surrogate objective**：
- `ratio * advantages`：未 clip 版本
- `clipped_ratio * advantages`：clip 版本
- 取 min：当 advantage > 0 时，ratio 太大（>1+ε）会被 clip 住，防止过度放大好回答；当 advantage < 0 时，ratio 太小（<1-ε）也会被 clip 住
- `compute_gen_logps=False` 时退化为无 clip 的简单 REINFORCE（不推荐）

```python
    per_token_loss = -(per_token_loss - beta * per_token_kl)
    # loss = -(surrogate_objective - β * KL_penalty)
    # 要最小化 loss = 最大化 objective - 最小化 KL 散度

    loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
    # 先对 answer 的 token 取平均（per-sequence mean），再对 batch 取平均
```

**最终归一化**：先除以每个样本的有效 token 数（`completion_mask.sum`），再对 batch 取平均。这样长回答和短回答对 loss 的贡献是公平的。

### 3.3 Reward 函数 —— `grpo_vllm_one.py:144-157`

```python
def reward_correct(item, answer):
    # 用正则提取回答中最后一个数字/分数/小数
    pattern = r'\d+\.\d+|\d+/\d+|\d+'
    nums = re.findall(pattern, answer)
    if len(nums) == 0: return -1.0      # 没找到数字 → 严厉惩罚
    lastnum = nums[-1]
    ans = parse(lastnum, extraction_config=[ExprExtractionConfig()])
    ground_truth = parse(item["A"], extraction_config=[ExprExtractionConfig()])
    return 1 if verify(ans, ground_truth) else -1   # 对了 +1，错了 -1

def reward_format(item, answer):
    # 检查格式: 必须是 思考过程 <answer>答案</answer>
    pattern = r"^💭.*?💭[\n ]*<answer>.*?</answer>$"
    think_count = answer.count("💭") + answer.count("💭")
    answer_count = answer.count("<answer>") + answer.count("</answer>")
    return 1.25 if re.match(pattern, answer, re.DOTALL | re.DOTALL) \
           and think_count == 2 and answer_count == 2 else -1
```

**总 reward = reward_correct + reward_format**：
| 情况 | correct | format | 总 reward |
|---|---|---|---|
| 都对 | +1 | +1.25 | **+2.25** |
| 答对但格式错 | +1 | -1 | **0** |
| 答错但格式对 | -1 | +1.25 | **+0.25** |
| 都错 | -1 | -1 | **-2** |

设计意图：**格式正确的权重比答案正确更高**（1.25 vs 1.0），在训练初期优先教会模型输出格式，后期再通过答案正确性来提升推理质量。

### 3.4 gen_worker 的关键逻辑 —— `grpo_vllm_one.py:160-229`

**组内归一化 + 跳过无信号组**：
```python
if curr_rewards.max() - curr_rewards.min() < 1e-4: continue  # 方差太小，跳过
curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-4)
```
这是 GRPO 的核心思想：如果一个组的所有回答 reward 一样（比如全对或全错），没有 advantage 信号，训练就没法区分好坏 → 跳过。

**计算 gen_logps（生成时的 log 概率）**：
```python
if compute_gen_logps:
    # 用 vLLM 的 prompt_logprobs 模式：temperature=0, max_tokens=1
    # 实际上不生成新 token，只是获取每个已有 token 在生成时的 log 概率
    gen_logps_sp = SamplingParams(temperature=0, top_p=1, max_tokens=1, prompt_logprobs=1)
    zz = vllm_gen.generate(prompt_token_ids=merged_ids.tolist(), sampling_params=gen_logps_sp)
    gen_logps = torch.tensor([[list(x.values())[0].logprob for x in xx] for xx in zz])
```
这是一个巧妙的 hack：让 vLLM 不生成任何新 token（`max_tokens=1`），只记录每个已有 token 的 log 概率。这些 logps 就是 $\pi_{\text{old}}$，用于计算 importance ratio。

**热更新 vLLM 权重**：
```python
def try_update_model():
    new_state_dict = Q.get_nowait()
    llm_model = vllm_gen.llm_engine.model_executor.driver_worker.model_runner.model
    llm_model.load_weights(new_state_dict.items())
```
每 `gen_update_steps` 步，训练进程把最新权重推到 `mp.Queue`，gen_worker 拉取后直接替换 vLLM 引擎里的模型权重。这样生成器用的策略会逐步跟上训练策略。

### 3.5 ref_server 的工作 —— `ref_server.py`

```python
while True:
    d = raw_queue.get()           # 从 /upload 接口收到原始 batch
    prompt_length = d['base']['plen']
    with torch.inference_mode():
        per_token_logps = get_per_token_logps(d['inputs'].to(ref_model.device))  # 算 ref logps
    per_token_logps = per_token_logps[:, prompt_length-1:]  # 只取 answer 部分
    # 拼上之前的 inputs/rewards/gen_logps，放 result_queue
    result_queue.put(xdata)
```

ref_server 的工作非常简单：
1. 收到 gen_worker POST 来的 batch（token_ids + rewards + gen_logps）
2. 用冻结的 ref model 算 per-token log 概率
3. 把 ref_logps 拼进去，放 result_queue
4. 训练进程 GET 时直接返回

**ref_server 的核心价值**：它持有唯一的、不被训练更新的一份 ref model 副本。如果把它放在训练进程里，DeepSpeed ZeRO-2 会在每个 GPU 上都复制一份，浪费显存。

### 3.6 序列化协议 —— `ref_server.py:5-25`

进程间不能用 Python 对象直接传，需要序列化成 bytes：
```python
# 单个 tensor → bytes
def tensor_to_bytes(t):
    buffer = io.BytesIO()
    torch.save(t, buffer)
    return buffer.getvalue()

# bytes → tensor
def bytes_to_tensor(b):
    return torch.load(io.BytesIO(b), weights_only=True)

# 多个 bytes 拼成一个包（带长度头）
def make_bytes_list(blist):
    buffer = io.BytesIO()
    buffer.write(len(blist).to_bytes(4, 'big'))       # 包含多少个元素
    for b in blist:
        buffer.write(len(b).to_bytes(4, 'big'))       # 每个元素的长度
        buffer.write(b)
    return buffer.getvalue()
```

每次上传的 batch 是一个 bytes_list，包含：`[JSON元数据, token_ids, rewards, gen_logps?]`。下载时 ref_server 追加了 `ref_logps`。

### 3.7 训练主循环 —— `grpo_vllm_one.py:232-315`

```python
# 启动三进程
if dist.get_rank() == 0:
    Q = mp.Queue()                    # 权重同步队列
    info_Q = mp.Queue()               # 生成信息队列（reward 统计等）
    p = mp.Process(target=gen_worker, args=(Q, gen_device, info_Q))
    p.start()

# DeepSpeed 初始化
engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model)

# 训练循环
for step in progress:
    batch = get_batch()               # 从 ref_server GET 完整 batch
    while batch is None:              # 如果 gen_worker 还没产出，等
        print('waiting for batch...')
        batch = get_batch()
    loss, avg_kl = GRPO_step(batch)
    engine.backward(loss)
    engine.step()

    # 每 gen_update_steps 步同步权重给 gen_worker
    if step % gen_update_steps == 0:
        state_dict = engine.module.state_dict()
        Q.put(state_dict)

    # 每 save_steps 步存档
    if step % save_steps == 0:
        engine.module.save_pretrained(save_name)
```

注意 `ds_config` 里的关键参数：
```python
ds_config = {
    "train_micro_batch_size_per_gpu": 1,        # 每个 GPU 每个 micro batch = 1
    "gradient_accumulation_steps": 16,           # 累积 16 步才更新
    "optimizer": {"type": "AdamW", "params": {"lr": 1e-6}},
    "bf16": {"enabled": True},                   # bf16 混合精度
    "zero_optimization": {"stage": 2, ...}        # ZeRO-2，分片优化器状态
}
```

**实际训练的 batch_size** = `train_micro_batch_size_per_gpu × gradient_accumulation_steps × num_gpus`。如果 2 个训练 GPU，每步实际处理了 2×16 = 32 个样本。

### 3.8 DeepSpeed 配置详解 —— `grpo_vllm_one.py:32-51`

| 参数 | 值 | 含义 |
|---|---|---|
| `train_micro_batch_size_per_gpu` | 1 | 每个 GPU 每个 micro batch 的样本数 |
| `gradient_accumulation_steps` | 16 | 累积 16 个 micro batch 才做一次 optimizer step |
| `optimizer.type` | AdamW | 优化器 |
| `optimizer.params.lr` | 1e-6 | 学习率。RL 微调通常比 SFT 小很多（SFT 一般 1e-5 ~ 2e-5） |
| `bf16.enabled` | True | 用 bf16 混合精度，省显存 + 加速 |
| `zero_optimization.stage` | 2 | ZeRO-2：分片优化器状态和梯度。比 ZeRO-3 省 communication 开销 |
| `offload_optimizer.device` | cpu | 优化器状态放 CPU，进一步省 GPU 显存 |

---

## 4. 环境、数据与运行

### 4.1 硬件需求

- **最少 2 GPU**：1 个训练 + 1 个 ref_server/gen_worker
- **推荐 3+ GPU**：2 个训练（DeepSpeed 多卡加速）+ 1 个 ref_server + gen_worker 可以共用训练之外的 GPU
- 本项目使用 Qwen2.5-3B（bf16 约 6GB），单 GPU 显存要求不高
- 机器环境：8×3090（每卡 24GB），详见 [[env-baseline-grpo]]

### 4.2 环境搭建

```bash
# 推荐克隆 ImBD 环境（torch 2.7 + CUDA 12.8）
conda create -n grpo --clone ImBD
conda activate grpo
cd simple_GRPO
pip install -r requirements.txt   # vllm==0.10.1.1, deepspeed, torch, transformers, bottle, tornado, math-verify

# GSM8K 数据需要 datasets 包（不在 requirements 里，需要额外装）
pip install datasets
```

### 4.3 数据

本项目默认用 **GSM8K**（小学数学题），通过 `datasets` 库在线加载：
```python
from datasets import load_dataset
dataset = load_dataset("openai/gsm8k", "main", split="train")
QAs = [{'Q': x, 'A': y.split('####')[-1].strip()} for x, y in zip(dataset['question'], dataset['answer'])]
```

`A` 字段取的是 `####` 后面的最终答案（如 `"42"`），用于 `reward_correct` 做比对。

**不装 `datasets` 会怎样？** 生成器不会产出任何数据，训练循环永远卡在 `waiting for batch...`。

REINFORCE++ 变体用的是 **MATH** 数据集（更难的数学竞赛题），从本地 JSON 文件加载，路径硬编码在代码里。

### 4.4 模型

默认模型路径 `./models/Qwen/Qwen2.5-3B`，需要提前下载。也可以换成任何 HuggingFace 上的 causal LM（修改 `model_path` 即可）。

### 4.5 从头训练：实操步骤

#### Step 0 — 确认 GPU 分配

```bash
# 查看哪些 GPU 空闲
nvidia-smi
```

假设有 8 卡（GPU 0-7），计划：
- GPU 7：ref_server（独立终端）
- GPU 2-6：训练进程（5 卡 DeepSpeed）
- GPU 2：gen_worker（从训练进程的 CUDA_VISIBLE_DEVICES 之外找——注意这里有个坑，见 §8）

**关键**：`gen_device = 3` 指的是训练进程的 `CUDA_VISIBLE_DEVICES` 中的**相对索引**。如果 `CUDA_VISIBLE_DEVICES=2,3,4,5,6`，那 `gen_device=3` 实际对应物理 GPU 5。所以确保 gen_worker 用的物理 GPU 不在训练进程的可见设备中。

#### Step 1 — 启动 ref_server（终端 1）

```bash
CUDA_VISIBLE_DEVICES=7 python ref_server.py
# 输出：Bottle v0.13 serving on http://0.0.0.0:59875/
```

#### Step 2 — 启动训练（终端 2）

```bash
CUDA_VISIBLE_DEVICES=2,3,4,5,6 deepspeed grpo_vllm_one.py
```

`gen_device=3` 意味着 gen_worker 会在物理 GPU 5 上运行 vLLM。

#### Step 3 — 观察训练

训练开始后，终端 2 会交替显示两类日志：
- **生成器日志**（gen_worker 进程）：`time: X.XXs rewards: [2.25, -2, ...] answers: 💭...`
- **训练器日志**（主进程）：`Loss: X.XXXXXX`
- SwanLab 监控面板：`swanlab watch` 查看 loss、KL、reward 曲线

#### Step 4 — 等待训练完成

默认 `all_steps=1000`，每 `gen_update_steps=16` 步同步一次权重，每 `save_steps=200` 步存档一次。

存档位置：`./step_200/`、`./step_400/`、`...`、`./step_1000/`。

---

## 5. 可调参数清单

### 5.1 脚本顶部参数（改代码即生效）

| 参数 | 默认 | 位置 | 含义 / GRPO 视角的注意点 |
|---|---|---|---|
| `model_path` | `"./models/Qwen/Qwen2.5-3B"` | line 12 | 基座模型路径 |
| `gen_device` | `3` | line 13 | gen_worker 使用的 GPU 索引（**相对**于 `CUDA_VISIBLE_DEVICES`） |
| `beta` | `0.04` | line 14 | **KL 惩罚系数**。越大→越保守（不离 ref 太远），越小→越激进。0.04 是偏温和的值 |
| `all_steps` | `1000` | line 15 | 总训练步数 |
| `Q_batch_size` | `5` | line 16 | 每轮生成抽几道题。增大→更多并行、但显存更高 |
| `num_pre_Q` | `8` | line 17 | 每题生成几个回答。这是 group 的大小，GRPO 的核心参数 |
| `train_batch_size` | `1` | line 18 | 每次 gen_worker POST 给 ref_server 的 batch size |
| `gen_update_steps` | `16` | line 19 | 每多少步同步一次权重给 gen_worker。越小→gen 用的新策略越及时，但通信开销大 |
| `save_steps` | `200` | line 20 | 存档间隔 |
| `compute_gen_logps` | `True` | line 21 | 是否计算生成时的 logps（用于 importance ratio clipping）。**False 会退化为无 clip 的简单 REINFORCE** |
| `clip_param` | `0.2` | line 22 | PPO clip 范围 [1-ε, 1+ε]。0.2 是 PPO 标准值 |
| `ref_server` | `"http://localhost:59875"` | line 23 | ref_server 地址 |

### 5.2 DeepSpeed 配置（`ds_config` dict）

| 参数 | 默认 | 含义 |
|---|---|---|
| `gradient_accumulation_steps` | `16` | 累积步数。影响实际 batch size |
| `optimizer.params.lr` | `1e-6` | 学习率。RL 微调比 SFT 小 10 倍量级 |
| `bf16.enabled` | `True` | bf16 混合精度 |
| `zero_optimization.stage` | `2` | ZeRO-2（不要改成 3，ZeRO-3 的 communication 开销大） |
| `offload_optimizer.device` | `cpu` | 优化器放 CPU，省 GPU 显存 |

### 5.3 生成参数（`gen_worker` 内）

| 参数 | 默认 | 位置 | 含义 |
|---|---|---|---|
| `temperature` | `0.9` | `SamplingParams` | 生成温度。0.9 偏探索，让回答更多样 |
| `max_tokens` | `700` | `SamplingParams` | 每个回答最大 token 数 |

### 5.4 Reward 函数参数（硬编码）

| 参数 | 值 | 含义 |
|---|---|---|
| 正确答案 reward | `+1.0` | `reward_correct` |
| 错误答案 reward | `-1.0` | `reward_correct` |
| 格式正确 reward | `+1.25` | `reward_format`（比正确性权重略高） |
| 格式错误 reward | `-1.0` | `reward_format` |

### 5.5 调参注意事项（学 GRPO 最该记住的几点）

1. **`num_pre_Q`（group size）是 GRPO 的灵魂**。太小（如 2-4）→ advantage 方差大、不稳定；太大（如 16-32）→ 生成慢、显存高。8 是经验值。
2. **`beta`（KL 系数）控制探索-利用平衡**。太大学不到东西（太保守），太小容易 reward hacking（格式对了但推理是乱来的）。
3. **`gen_update_steps` 影响策略滞后**。gen_worker 用的策略比训练进程落后若干步。太小（1-2）→ 通信开销大；太大（64+）→ gen_worker 的策略太旧，ratio 偏离大，clip 频繁触发。
4. **`lr` 要比 SFT 小一个量级**。RL 的梯度噪声大，1e-6 是安全起点。太大容易 loss 爆炸。
5. **先格式后内容**：初期 `reward_format` 占主导（因为随机初始化的模型很难答对数学题），训练中后期 `reward_correct` 逐渐起作用。观察 SwanLab 的 `reward/mean` 曲线可以看到这个转折。
6. **`compute_gen_logps` 必须为 True**。否则没有 importance ratio clipping，等价于无约束的策略更新，容易不稳定。

---

## 6. 动手调参建议（把 GRPO 玩明白）

### 6.1 冒烟测试（5 分钟确认流程通）

```python
# 修改参数
all_steps = 10
Q_batch_size = 2
num_pre_Q = 4
save_steps = 5
gen_update_steps = 4
```

确认三进程启动、batch 生成、loss 下降、存档正常。

### 6.2 beta 扫描

| 实验 | beta | 预期 |
|---|---|---|
| A | 0.001 | 激进，KL 约束弱，模型可能学会投机取巧 |
| B | 0.04（默认） | 平衡 |
| C | 0.1 | 保守，模型变化慢，但更稳定 |
| D | 0.0（去掉 KL） | 对比基线，看 KL 惩罚的必要性 |

### 6.3 group size 消融

| 实验 | num_pre_Q | 观察 |
|---|---|---|
| A | 2 | advantage 方差大，训练不稳定 |
| B | 4 | 较稳定 |
| C | 8（默认） | 平衡 |
| D | 16 | 更稳定但生成更慢 |

### 6.4 reward 设计实验

- **去掉 format reward**：只留 `reward_correct`，看模型能不能自己学会格式
- **交换权重**：`correct=1.25, format=1.0`，看是否影响训练动态
- **三元 reward**：`correct=2, format_only=0.5, wrong=-2`（REINFORCE++ 的设计），对比效果

### 6.5 关闭 clipping 的对照

设 `compute_gen_logps = False`，观察：
- loss 曲线是否更震荡
- KL 散度是否增长更快
- 最终准确率是否下降

### 6.6 学习率扫描

| 实验 | lr | 观察 |
|---|---|---|
| A | 1e-7 | 可能太慢，1000 步学不够 |
| B | 1e-6（默认） | 基线 |
| C | 5e-6 | 可能够用但要注意稳定性 |
| D | 1e-5 | RL 里偏大，可能 loss 爆炸 |

---

## 7. 四个变体对照（理解 GRPO 在 RL 家族中的位置）

| 目录 | 算法 | Loss 特点 | 数据集 | 特殊功能 |
|---|---|---|---|---|
| **root** | GRPO | KL 惩罚 + clipped ratio | GSM8K | SwanLab 日志 |
| **simple_grpo_v1/** | GRPO | 同上 | GSM8K | 可选 triton kernel 加速 |
| **simple-reinforce++/** | REINFORCE++ | **无 KL 惩罚**，per-token advantage，`num_items_in_batch` 归一化 | MATH | per-token advantage |
| **Auto_Program/** | GRPO | KL 惩罚 + clipped ratio | MATH | 代码执行、动态 reward 权重、WandB |

### 7.1 GRPO vs REINFORCE++ 的核心差异

**GRPO loss**（本项目 root）：
```python
per_token_loss = min(ratio * advantages, clipped_ratio * advantages)
loss = -(per_token_loss - beta * KL)  # 有 KL 惩罚
loss = loss.mean_per_sequence.mean_per_batch  # 先 per-sequence 均值，再 batch 均值
```

**REINFORCE++ loss**（`simple-reinforce++/`）：
```python
per_token_loss = -min(ratio * advantages, clipped_ratio * advantages)
loss = loss.sum() / num_items_in_batch  # 所有 token 求和除以总 token 数
# 无 KL 惩罚
```

关键区别：
1. **REINFORCE++ 没有 KL 惩罚**——它不需要 ref_server 算 ref_logps（但本项目的 ref_server 仍然参与了数据的传递）
2. **归一化方式不同**——GRPO 先对每个样本的 token 取均值再 batch 均值；REINFORCE++ 直接除以所有 token 总数
3. **advantage 来源不同**——REINFORCE++ 的 advantage 在 gen_worker 里就做成了 per-token 形式（通过 `gen_logps` 的加权）

### 7.2 Auto_Program 的特色

- **动态 reward 权重**：训练初期重格式（`2*format`），16 步后重准确率（`2*acc`），逐步从「学会格式」过渡到「学会推理」
- **Python 代码执行**：生成的回答可以包含 Python 代码，`call_python_score` 会真的执行并验证结果
- **WandB 日志**：记录到 WandB 而不是 SwanLab

---

## 8. 常见坑

| 现象 | 原因 | 处理 |
|---|---|---|
| `waiting for batch...` 一直卡住 | 没装 `datasets` / 网络不通 / gen_worker 崩了 | `pip install datasets`；检查 gen_worker 终端有没有报错 |
| gen_worker 启动后立刻死 | GPU 冲突：gen_device 指向的物理 GPU 正在被其他进程占用 | `nvidia-smi` 确认 GPU 空闲；注意 `gen_device` 是相对索引 |
| ref_server 返回 `b'string'` | ref_server 版本和训练脚本不匹配（len(dd) 判断） | 确保用同目录的 `ref_server.py` |
| loss 是 NaN | lr 太大 / bf16 溢出 / reward 全 0 | 降 lr 到 1e-7；检查 reward 是否有正有负 |
| loss 不下降 | KL 惩罚太强（beta 太大）/ lr 太小 / reward 信号太弱 | 降 beta、升 lr、检查 `reward/mean` 曲线 |
| OOM | ZeRO stage 太低 / batch 太大 / gen_worker GPU 和训练重叠 | 升到 ZeRO-2 + offload CPU；减小 `Q_batch_size` |
| SwanLab 报错 | 网络问题或未登录 | 可以注释掉 `swanlab.init` 和 `swanlab.log` |
| vLLM weight load 失败 | gen_worker 和训练进程的 `model_path` 不一致 / DeepSpeed ZeRO-3 | 确保路径一致；用 ZeRO-2 |
| 训练初期 reward 全是负数 | 正常现象。随机初始化的模型很难同时答对数学题且格式正确 | 等几十步后会开始出现正 reward |

---

## 9. 关键概念速查表

| 概念 | 代码里的名字 | 通俗理解 |
|---|---|---|
| Policy model ($\pi_\theta$) | `engine` (DeepSpeed) | 当前正在训练的模型 |
| Reference model ($\pi_{\text{ref}}$) | ref_server 里的 `ref_model` | 冻结的初始模型，提供 KL 锚点 |
| Old policy ($\pi_{\text{old}}$) | `gen_logps` | 生成回答时的模型快照，用于计算 importance ratio |
| Advantage ($A$) | `batch['rewards']`（已归一化） | 组内相对好坏，正=好、负=差 |
| Importance ratio ($\rho$) | `ratio = exp(logps - gen_logps)` | 当前策略 vs 生成时的策略偏离了多少 |
| Clipping | `clipped_ratio = clamp(ratio, 1-ε, 1+ε)` | 防止策略一步更新太大 |
| KL penalty | `per_token_kl = exp(ref - logps) - (ref - logps) - 1` | 防止模型偏离初始策略太远 |
| Group | 同一道题的 `num_pre_Q` 个回答 | GRPO 的核心：组内竞争产生 advantage |
| Completion mask | `inputs[:, prompt_length:] != pad_token_id` | 只优化回答部分，不动 prompt |

---

## 10. 关键文件速查

```
simple_GRPO/
  grpo_vllm_one.py          # ← 主 GRPO 实现（~315 行），先读这个
  ref_server.py             # ← 参考模型 HTTP 服务（~92 行），第二读
  requirements.txt          # 依赖版本

  simple_grpo_v1/
    grpo_ref_split.py       # 早期版本，无 vLLM，可选 triton kernel
    ref_server.py           # 同架构，端口 59875

  simple-reinforce++/
    rf++_vllm_one.py        # REINFORCE++ 变体（无 KL，不同归一化）
    config.py               # 统一配置（base_config + ds_config + train_config）
    ref_server.py           # 端口可配

  Auto_Program/
    hjy_grpo_program.py     # GRPO + 代码执行 + 动态 reward
    config.py               # 同上
    ref_server.py           # 同上
    prompts/                # system prompt .txt 文件
```

### 推荐学习路线

1. **入门**：只看 `grpo_vllm_one.py` + `ref_server.py`，跑通 GSM8K 训练
2. **对比**：看 `simple-reinforce++/rf++_vllm_one.py` 的 `REINFORCE_plusplus_step`，和 GRPO loss 逐行对比
3. **进阶**：看 `Auto_Program/hjy_grpo_program.py` 的 reward 设计和动态权重
4. **工程**：研究三进程的 GPU 分配和序列化协议
