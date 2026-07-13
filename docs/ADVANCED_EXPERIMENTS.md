# 进阶实验建议（Advanced Experiments Guide）

> 面向：已经跑通基础训练、做完了 `GRPO_LEARNING_GUIDE.md §6` 里那些"调一调 group size / beta / reward 开关"的读者。
> 定位：本文是 §6 的**进阶续作**——不再讲"动一动超参"，而是讲 reward 工程、优化器替换、算法层改造（DAPO / 熵 / 课程）、工具调用迁移，以及把它们横向对比所需的评测脚手架。
> 阅读建议：先看 §0（评测脚手架，所有实验的可比性前提）→ 再按你的兴趣跳到 §1 reward / §2 优化器 / §5 工具调用 → 最后用 §8 的实验路线图串起来。
> 配套代码：所有改动都以 root 目录的 `grpo_vllm_one.py`（行号对应当前版本）为基准；其它变体结构类似。

---

## 0. 先决条件：一个干净的评测脚手架（最重要的一件事）

在动任何"进阶"念头之前，先补上目前最大的盲区：**项目只在训练时记录 on-policy 的 `reward_acc`（带 temperature=0.9 采样噪声），没有任何 held-out 评测。** 没有它，"换 Momentum 后格式准确率涨了"这种结论没法下——你分不清是改动起效还是采样方差。

### 0.1 要测什么

| 指标 | 怎么算 | 为什么 |
|---|---|---|
| `eval/acc` | GSM8K test split，greedy(temperature=0)，`math_verify` 比对 | 真正的"答对率"，去噪 |
| `eval/format_acc` | greedy 输出匹配 `^<think>.*?</think>[\n ]*<answer>.*?</answer>$` | 格式合规率 |
| `eval/format_but_wrong` | format 对但 acc 错的比例 | 监控 reward hacking |
| `eval/pass_at_k` | 同一题采 k=8 个，至少一个对的占比 | 反映能力上限，比 greedy acc 更稳 |
| `avg_completion_len` | 平均回答 token 数 | 监控 degeneration / 冗长化 |

### 0.2 最小实现：独立评测脚本（不动训练代码）

因为 checkpoint 已经按 `{save_dir}/{run_tag}/step_{step}` 周期性保存，写一个独立脚本 `eval_ckpt.py` 最干净——可以离线批量评所有 step，互不干扰训练：

```python
# eval_ckpt.py（草稿，~60 行）
# 用法: CUDA_VISIBLE_DEVICES=0 python eval_ckpt.py ckpts/<run_tag>/step_1000
import torch, re, sys, json
from transformers import AutoTokenizer, AutoModelForCausalLM
from vllm import LLM, SamplingParams
from datasets import load_dataset
from math_verify import parse, verify, ExprExtractionConfig

ckpt = sys.argv[1]
tok = AutoTokenizer.from_pretrained(ckpt)
llm  = LLM(model=ckpt, gpu_memory_utilization=0.6)
ds   = load_dataset("openai/gsm8k", "main", split="test")      # 注意是 test split
sysp = """You are a helpful assistant. ... <think>...</think><answer>...</answer> ..."""  # 复用训练用的 system_prompt

greedy = SamplingParams(temperature=0, max_tokens=700)
prompts = [tok.apply_chat_template([{"role":"system","content":sysp},{"role":"user","content":q}],
                                  tokenize=False, add_generation_prompt=True) for q in ds['question']]
outs = llm.generate(prompts, greedy, use_tqdm=True)

def acc(ans, gt):
    nums = re.findall(r'\d+\.\d+|\d+/\d+|\d+', ans)
    if not nums: return 0
    return int(verify(parse(nums[-1], extraction_config=[ExprExtractionConfig()]),
                      parse(gt, extraction_config=[ExprExtractionConfig()])))
fmt_re = re.compile(r"^<think>.*?</think>[\n ]*<answer>.*?</answer>$", re.DOTALL)
gts = [a.split('####')[-1].strip() for a in ds['answer']]

n = len(outs); a = sum(acc(o.outputs[0].text, gts[i]) for i,o in enumerate(outs))
f = sum(bool(fmt_re.match(o.outputs[0].text)) for o in outs)
print(json.dumps({"ckpt":ckpt,"eval_acc":a/n,"eval_format_acc":f/n}, indent=2))
```

> 关键：训练用 `train` split、评测用 `test` split，二者绝不重叠。当前训练代码 `gen_worker` 里 `load_dataset("openai/gsm8k","main",split="train")` 只用了 train，所以 test 是干净的。

### 0.3 想要更省事：训练进程内嵌周期性评测

如果不想离线跑，可以在主循环 `if step % eval_steps == 0` 时，**由 rank0 用一个额外的 vLLM 实例**（或临时复用 gen_worker）对 test 子集做 greedy 评测并 `swanlab.log`。但跨进程复用 vLLM 较麻烦，**首选还是 §0.2 的独立脚本**——它还能评历史 checkpoint，画出"acc 随 step 变化"曲线，对判断"格式 reward 该何时衰减"至关重要（见 §1.2）。

> ✅ **建议把"补 §0 的评测脚本"作为第 0 个实验先做掉。** 下面所有实验都假设你有了它。

---

## 1. Reward 工程进阶（你的核心兴趣）

当前 reward 配置在 `grpo_vllm_one.py:27-33`，总 reward 在 `gen_samples`（`grpo_vllm_one.py:247-254`）里逐条累加，然后组内归一化（`grpo_vllm_one.py:298`）。

### 1.1 交换 correct / format 权重

你已经计划把 `reward_correct_right=1.25 / reward_format_right=1.0`（当前是 1.0 / 1.25）。改这一组变量即可，`build_run_tag()` 会自动把 run_tag 从 `..._cor1+fmt1.25` 变成 `..._cor1.25+fmt1`，ckpt 与 swanlab 都不冲突。

**先把整张 reward 表算清楚再跑**（`reward_*_wrong` 仍是 -1.0）：

| 情况 | 当前 (cor=1.0, fmt=1.25) | 交换后 (cor=1.25, fmt=1.0) |
|---|---|---|
| 都对 | +2.25 | +2.25 |
| 答对、格式错 | 1.0 − 1.0 = **0.0** | 1.25 − 1.0 = **+0.25** |
| 答错、格式对 | −1.0 + 1.25 = **+0.25** | −1.0 + 1.0 = **0.0** |
| 都错 | −2.0 | −2.0 |

交换的真正效果是**翻转中间两行的排序**：现在"格式漂亮但算错"（+0.25）比"算对但格式丑"（0.0）分高；交换后反过来。也就是把价值从"先学格式"挪向"先学正确性"。

**预期与风险**：
- 早期模型几乎算不对数学题，format 又被相对压低 → 组内更难出现"有梯度的混合组"，可能**起步更慢**。务必盯 `eval/format_acc`：如果它迟迟不上来（比如 200 步还 <0.3），说明 format 失去了"脚手架"作用，这时要么回到原配比，要么配合 §1.2 的动态权重。
- 这是一个干净的 A/B，**必须**和 baseline（cor=1.0/fmt=1.25）用同一个 seed（见 §7.1）、同一个评测脚本（§0）对比 `eval/acc` 曲线。

### 1.2 自适应动态 reward（format_acc 触发衰减）——你的旗舰想法，但有个关键洞察

你的设想：当 format 累计准确率 > 0.8 时，调小 format reward，让信号更聚焦到正确性。方向对，但先理解一个会让它"事倍功半"的事实：

> ⚠️ **关键洞察：GRPO 的组内归一化已经自动把"已掌握的 reward"归零了。**
>
> advantage ∝ r_i − mean(r)，而 r_i = w_c·c_i + w_f·f_i。
> 代入：A_i ∝ w_c·(c_i − c̄) + w_f·(f_i − f̄)。
> 当一个 group 里 8 条回答**格式全对**时，f_i − f̄ = 0，于是 **w_f 这一项对梯度贡献恰好为 0，无论 w_f 多大**。
> 全局 format_acc → 1 时，绝大多数 group 都是"格式全对"的均匀组 → format 权重早已**自动失效**。

**推论**：到 format 真正饱和时再衰减 w_f，效果很弱（它本来就接近 0 了）。动态衰减**真正起作用的区间是过渡带（format_acc ≈ 0.5–0.95）**，它能：(a) 加速"注意力"从格式转向正确性；(b) 防止少数"混合格式"组过度奖励格式。所以这个实验值得做，但要**在过渡带观察**、并配合 §1.1 的交换 / §1.3 的拆分才看得到明显收益。

**实现要点**（都在 `gen_worker` 里，因为 reward 在那里算）：

1. **指标要在 gen_worker 内部维护**（不要从训练进程往回传）。当前 `info_Q.put({...})`（`grpo_vllm_one.py:281-283`）只把统计往外送用于日志；自适应控制需要 EMA 留在 gen_worker 自己手里。

2. **用 EMA 而非瞬时值**，否则在阈值附近反复震荡：
   ```python
   # gen_worker 顶部
   fmt_acc_ema = 0.5   # 初始先验
   ```
   ```python
   # gen_samples 里，算完 format_score 后
   batch_fmt_acc = sum(f > 0 for f in format_flags_this_batch) / len(format_flags_this_batch)
   fmt_acc_ema = 0.95 * fmt_acc_ema + 0.05 * batch_fmt_acc
   ```

3. **用平滑 schedule，别用硬阈值**。硬阈值（>0.8 就降）会震荡：降了→format_acc 掉→又升回。用连续衰减 + 地板：
   ```python
   # format_acc ≤ 0.6 时给满额 1.25；>0.6 后线性衰减到地板 0.3
   w_fmt = reward_format_right * max(0.3, 1.0 - (fmt_acc_ema - 0.6) / 0.4)
   ```
   想要"触发式"也可以，但用**双阈值滞回**（>0.85 降到地板，<0.6 才恢复满额）。

4. **可选：同步上调 w_correct**，保持总 reward scale 大致不变（归一化下不那么关键，但动力学更稳）。

5. **必须把当前 w_fmt / fmt_acc_ema 也 log 到 swanlab**（通过 info_Q 传出来），否则没法解释曲线。

6. **对照**：跑三条——(a) 静态 baseline、(b) §1.1 静态交换、(c) 动态衰减。在 §0 的评测脚本里看 `eval/acc` 和 `eval/format_acc` 谁先上、谁的天花板高。

> 📌 这个思路和 `Auto_Program/hjy_grpo_program.py:238-242` 的"按 step 切换权重"是同一类 curriculum，但**触发量不同**：Auto_Program 用固定 step（`update_model_num >= 16`），你用**性能指标**——后者更鲁棒，因为不同 seed/模型达到"格式学会"的步数不一样，按 step 切容易切早或切晚。

### 1.3 拆分双 advantage（更彻底的解法，进阶）

§1.2 的洞察暴露了一个更本质的问题：把 correct 和 format 加成一个标量再一起归一化，会**互相稀释**。一个 group 如果"格式全对、正确性参半"，format 信号为 0，只剩 correct 信号——这没问题；但反过来"正确性全对、格式参半"时 correct 信号也被吃掉。

**更干净的做法**：把两个 reward 当作独立目标，各自在 group 内归一化、各自算 surrogate loss、再相加：

```python
# 伪代码，GRPO_step 改造方向
A_correct = group_normalize(reward_correct_per_ans)   # (B,1)
A_format  = group_normalize(reward_format_per_ans)    # (B,1)
# 各自跑一遍 clipped surrogate
surrogate = -min(ratio*A_correct, clip*A_correct) - min(ratio*A_format, clip*A_format)
loss = (surrogate * mask).sum(1)/mask.sum(1).mean()
```

这要求 gen_worker 把 `correct_raw` 和 `format_raw` **分别**传过 ref_server（当前只传合并后的 `rewards`），ref_server 协议要加字段——工作量中等。`Auto_Program` 已经在传 `acc_scores`/`format_scores`（`hjy_grpo_program.py:337-338`），可以借鉴它的协议扩展。

**收益**：直接消除"format 学会了反而压制 correct 信号"的耦合，是治理 reward hacking 最对症的一招。优先级：先做完 §1.1/§1.2，如果还看到"格式 99% 但 acc 卡住"，再上这个。

### 1.4 其它 reward shaping 细节（低成本，逐个试）

- **correct 三值化**：当前"没提取到数字"和"数字算错"都给 −1.0（`grpo_vllm_one.py:228`），把"没数字"信号丢了。改成 `+1 正确 / −0.5 算错 / −1 没数字`，能在错误答案之间也产生梯度。
- **长度惩罚**：观察到回答越来越长（看 §0 的 `avg_completion_len`）时，加 `−λ · len`（λ 取 1e-3 量级）或对超过阈值的回答扣分，抑制冗长 degeneration。
- **格式严谨度**：现在格式检查允许 `<think>`/`<answer>` 之间任意内容，可加一个"answer 里必须含数字"的弱约束。

### 1.5 advantage 归一化的 std 下限（稳定性，容易忽略）

`grpo_vllm_one.py:298`：`(curr_rewards - mean) / (std + 1e-4)`。group=8 + 二值 reward 时，std 经常很小，1e-4 的 eps 偏小，**个别离群 advantage 会爆到几十**，造成尖刺更新。建议：

```python
curr_rewards = (curr_rewards - curr_rewards.mean()) / (curr_rewards.std() + 1e-2)  # eps 调大
# 或者更稳：直接对 advantage 做 clamp
```

trl 上游和一些复现会用更大的 eps 或对 std 取全局均值。这是个不起眼但能减少 loss 尖刺的旋钮，扫 reward 实验时建议先固定一个更稳的归一化。

---

## 2. 优化器：从 AdamW 换成 Muon（你的另一个兴趣）

当前 `ds_config`（`grpo_vllm_one.py:52-54`）：`AdamW, lr=1e-6`，无 weight_decay、无梯度裁剪。换 **Muon** 比"换个 config type"重得多——它没有 DeepSpeed 原生支持，需要自带 optimizer 实现 + 改 `deepspeed.initialize` 的调用方式。但它在 LLM 训练上的 track record（Kimi K2 等在规模上用过）比 vanilla SGD/Momentum 强很多，是个**值得做的真·进阶实验**。

### 2.1 先做便宜的对照（换 Muon 之前）

在动 Muon 这种大改动前，先在 AdamW 框架内试这几个，性价比更高，也能拿到"加强 baseline"用于后续对照：

- **LR schedule**：当前 lr 恒定。加 warmup + cosine：
  ```python
  "scheduler": {
      "type": "WarmupDecayLR",
      "params": {"warmup_min_lr": 0, "warmup_max_lr": 1e-6,
                 "warmup_num_steps": 20, "total_num_steps": all_steps, "warmup_type": "linear"}
  },
  ```
- **betas**：RL 梯度噪声大，把 `beta1` 从 0.9 提到 0.95（更平滑），有时比换优化器还管用。注意 DeepSpeed 的 AdamW config 里要显式写 `"betas": [0.9, 0.999]`。
- **decoupled weight decay**：当前虽是 AdamW 但没设 wd，等价于 Adam。给 `weight_decay: 0.01` 看 RL 微调会不会更稳。

### 2.2 Muon 是什么（先理清，不然实现会抄错）

**Muon = Momentum + Orthogonalization**（"Momentum" 在这里是 SGD-with-momentum 的 momentum buffer，**不是**你要换成的那个东西——别和优化器名字混了）。核心两步：

1. 维护一个 momentum buffer（和 SGD-momentum 一样）；
2. 对**动量更新方向**做一次正交化（用 Newton–Schulz 迭代近似 `UV^T`，避免真做 SVD），让每个 2D 权重的更新是一个正交矩阵（旋转/反射），再按 `0.2·sqrt(max(d_out, d_in))` 缩放，使步长对矩阵形状不变。

**关键约束：只对 2D 参数（`nn.Linear` 的 weight）用 Muon。** 1D / 其它形状（embedding、layer norm、bias、`lm_head`）**不能用**（正交化没定义），标准做法是给它们配一个 **AdamW 回退**。所以你最终要的是一个 **Muon + AdamW 混合优化器**，按参数组分发。

### 2.3 参考实现（直接抄，Newton–Schulz 5 步版）

```python
# muon.py —— 经典实现（Keller Jordan 等），系数 (3.4445, -4.7750, 2.0315)
@torch.no_grad()
def zeropower_via_newtonschulz5(G, steps=5, eps=1e-7):
    assert G.ndim == 2
    a, b, c = 3.4445, -4.7750, 2.0315
    X = G.to(torch.float32) / (G.norm() + eps)   # NS 迭代建议 fp32，bf16 容易飘
    if G.size(0) > G.size(1): X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if G.size(0) > G.size(1): X = X.T
    return X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, nesterov=True, ns_steps=5):
        super().__init__(params, dict(lr=lr, momentum=momentum, nesterov=nesterov, ns_steps=ns_steps))
    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            lr, mu, nes, ns = g['lr'], g['momentum'], g['nesterov'], g['ns_steps']
            for p in g['params']:
                if p.grad is None: continue
                st = self.state[p]
                if 'momentum_buffer' not in st:
                    st['momentum_buffer'] = torch.zeros_like(p.grad)
                buf = st['momentum_buffer']
                buf.mul_(mu).add_(p.grad)
                update = p.grad.add(buf, alpha=mu) if nes else buf
                if update.ndim >= 2:                 # 2D → 正交化
                    update = zeropower_via_newtonschulz5(update.reshape(len(update), -1), ns)
                    update *= max(1.0, update.size(0) / update.size(1)) ** 0.5
                p.add_(update, alpha=-lr)
```

> 系数有多种版本（上面是流传最广的 5 步版；社区有更新的 6 步系数 set，收敛略好）。NS 迭代**务必用 fp32**——bf16 下正交化会失稳，这是常见踩坑点。

### 2.4 参数组拆分（哪些给 Muon，哪些给 AdamW）

对 Qwen2.5-3B 这类 transformer：

| 参数 | 归属 | 理由 |
|---|---|---|
| `q_proj/k_proj/v_proj/o_proj` weight、`gate/up/down_proj` weight | **Muon** | 2D 矩阵，Muon 主战场 |
| `embed_tokens.weight` | AdamW | 语义上是查找表，通常排除在 Muon 外 |
| `lm_head.weight` | AdamW | 同上（若与 embedding tied 要特别小心） |
| `LayerNorm`/`RMSNorm` weight、所有 bias | AdamW | 1D，无法正交化 |

拆分代码思路：
```python
muon_params, adamw_params = [], []
for n, p in model.named_parameters():
    if p.ndim >= 2 and "embed" not in n and "lm_head" not in n:
        muon_params.append(p)
    else:
        adamw_params.append(p)
```
然后用一个**包装类**把两组塞进同一个 `step()`（DeepSpeed 只接收一个 optimizer）：

```python
class MuonWithAdamW:
    def __init__(self, muon_params, adamw_params, muon_lr, adam_lr):
        self.muon = Muon(muon_params, lr=muon_lr)
        self.adamw = torch.optim.AdamW(adamw_params, lr=adam_lr)
    def step(self): self.muon.step(); self.adamw.step()
    def zero_grad(self, set_to_none=True): self.muon.zero_grad(set_to_none); self.adamw.zero_grad(set_to_none)
    # state / param_groups / load_state_dict 按需转发给两者
```

### 2.5 DeepSpeed 集成（最需要小心的部分）

**Muon 不能走 `ds_config` 的 `"optimizer": {"type": ...}` 路径**——DeepSpeed 不认识它。必须手搓实例传给 `deepspeed.initialize`：

```python
# grpo_vllm_one.py:360 附近，把
#   engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model, model_parameters=...)
# 改成
optimizer = MuonWithAdamW(muon_params, adamw_params, muon_lr=1e-4, adam_lr=1e-6)
engine, optimizer, _, _ = deepspeed.initialize(config=ds_config, model=model,
                                               optimizer=optimizer)   # 注意传 optimizer 而非 model_parameters
```

> ⚠️ **代价：传自定义 optimizer 会让 DeepSpeed 放弃 ZeRO-1/2 的优化器状态分片**（它会告警，每个 rank 持有完整的 optimizer state）。本项目靠 `offload_optimizer: cpu` 把优化器状态放 CPU，加上 Muon 的 2D 参数只有 1 份 momentum buffer（比 Adam 的 2 份还省），3B 模型在 CPU offload 下是扛得住的。但**必须显式加梯度裁剪**：
> ```python
> "gradient_clipping": 1.0,   # ds_config 里加，RL 噪声大，Muon 无自适应缩放，易爆
> ```
> （注意 `simple-reinforce++/config.py` 里这行是注释掉的，可参考。）

LR schedule 在自定义 optimizer 下要自己挂（`torch.optim.lr_scheduler`），或继续用 DeepSpeed 的 scheduler 但确认它驱动的是 wrapper 暴露出来的 param_groups——这一块最容易出 bug，建议第一版先**用恒定 lr 跑通**再加 schedule。

### 2.6 lr 标定与预期（避免"训练看起来死了"或"一发炸"）

> ⚠️ **Muon 的 lr 量级和 Adam 完全不同，不要直接套 1e-6。**
>
> 经验：预训练里 Adam ~3e-4 时 Muon ~0.01–0.02（约 30–60 倍）。RL 微调这里 Adam=1e-6 极小，**Muon 大致要从 1e-4 量级起**，且需要扫。
> - **两组 lr 要分别设**：`muon_lr`（2D 矩阵）和 `adam_lr`（embedding/norm/head 回退）。常见配比 `adam_lr ≈ muon_lr / 10 ~ muon_lr / 50`。
> - 建议起点 `muon_lr=1e-4, adam_lr=3e-6`，扫 `muon_lr ∈ {3e-5, 1e-4, 3e-4, 1e-3}`。
> - 前 50 步盯 loss 是否真的在动、`eval/acc` 是否有变化；一动不动=lr 太小，loss 飞或 NaN=lr 太大或该加 clip/warmup。

**预期（先设好心理预期）**：
- Muon 在 **LLM 预训练**上证据充分（收敛更快、loss 更低），但**在 RL/PPO/GRPO 微调上的公开结果很少**——这正是这个实验的新颖之处。
- 可能的结果：收敛更快（每步有效信号更强）；但也可能因为 RL 梯度是"稀疏 + 高方差 + 只来自 completion token"而和 Muon 假设的"密集梯度"不匹配，表现不如 AdamW。两种结果都有发表价值。
- 想降低风险，可以先只在 **2D 矩阵的某子集**（比如只给 MLP 的 `down_proj`）上试 Muon，确认 pipeline 通了再铺开。

### 2.7 实验设计

| 实验 | optimizer | muon_lr / adam_lr | 观察 |
|---|---|---|---|
| baseline | AdamW | — (1e-6) | 基线 |
| AdamW+schedule | AdamW + WarmupDecayLR | — (1e-6) | schedule 增益 |
| Muon (主) | Muon+AdamW + clip | 1e-4 / 3e-6 | 主对照，盯前 50 步是否动 |
| Muon (lr 扫) | Muon+AdamW + clip | 3e-4 / 1e-5 | lr 扫描 |

四个都用 §0 评测脚本评 `eval/acc` 曲线，别只看训练 loss。**建议手动给 `run_tag` 赋值**（如 `"muon_lr1e-4"`），因为 `build_run_tag()` 只编码 lr、不编码优化器类型，会和 AdamW 实验混。

---

## 3. 算法层进阶（DAPO / 熵 / baseline）

### 3.1 DAPO 的 clip-higher（非对称 clip，高价值）

当前 `clip_param=0.2` 对称 clip（`grpo_vllm_one.py:139`）：ratio 被夹在 [0.8, 1.2]。DAPO（Qwen 团队）指出：对称 clip 会**抑制探索**——好回答想往上提概率，ratio 一过 1.2 就被砍，模型学不动"新行为"。修法是非对称（**clip-higher**）：

```python
clip_low, clip_high = 0.2, 0.28       # 上界放宽
clipped_ratio = torch.clamp(ratio, 1-clip_low, 1+clip_high)
```

这个改动小、风险低、对"GRPO 后期 acc 卡住/探索坍缩"特别有效，**强烈建议放进任意 baseline**。配套地，DAPO 还有几个本项目已经天然满足的点：
- **Dynamic sampling**（跳过 reward 方差为 0 的组）——本项目 `grpo_vllm_one.py:294` 已经做了 ✓
- **Token-level loss**（按 token 而非 sequence 归一化）——见 §3.3 / 可对比 REINFORCE++ 的 `num_items_in_batch` 归一化
- **Overlong reward shaping**——对超长输出给软惩罚，配合 §1.4 的长度惩罚

### 3.2 熵奖励（entropy bonus，防过早收敛）

当 reward 曲线平台化、模型输出越来越单一时，加一个熵项鼓励探索：

```python
# GRPO_step 里，logits 得到后
log_probs = logits.log_softmax(dim=-1)            # (B, L-1, V)
entropy = -(log_probs.exp() * log_probs).sum(-1)  # (B, L-1) 每 token 熵
entropy_bonus = (entropy * completion_mask).sum() / completion_mask.sum()
per_token_loss = -(per_token_loss - beta * per_token_kl) - entropy_coef * entropy_per_token
```

`entropy_coef` 取 1e-2 ~ 1e-3。本质是"在 reward 之外给多样性一个直接梯度"，缓解 GRPO 把策略压扁的问题。

### 3.3 跨组 baseline（向 REINFORCE++ 靠拢，可选）

GRPO 用 group 内均值做 baseline，group=8 时方差仍大。可以维护一个**跨组 moving-average baseline** b，advantage 改为 `r_i − b`（再叠 group 归一化）。这能进一步降方差，但会让算法偏离"纯 GRPO"，作为"GRPO vs REINFORCE++ 之间"的中间点来对照很有趣——代码可直接借鉴 `simple-reinforce++/rf++_vllm_one.py:52-72` 的 `REINFORCE_plusplus_step`。

---

## 4. 数据 / 课程学习（和动态 reward 是一对）

### 4.1 难度采样 / prioritized sampling（直接提高数据利用率）

当前 `random.sample(QAs, Q_batch_size)`（`grpo_vllm_one.py:275`）完全随机。但 GRPO 会**跳过** reward 方差为 0 的组（全对/全错），这些组纯粹浪费生成算力。

**思路**：在 gen_worker 维护每道题的近期 solve rate（滑动平均），**优先采样 solve rate ∈ [0.3, 0.7] 的题**——这些题最容易产生"有梯度的混合组"，直接降低 skip rate。

```python
# gen_worker 里维护
solve_rate = {q_idx: 0.5 for q_idx in range(len(QAs))}   # 先验
# 每次生成完，按 group 的 acc 更新对应题的 solve_rate（EMA）
# 采样时按 |solve_rate - 0.5| 升序、加一点随机性挑 Q_batch_size 道
```

**先量化再优化**：在 swanlab 加一个 `batch/skip_rate`（被跳过的 group 占比）。如果它 >30%，说明大量算力被浪费，§4.1 的收益就很可观；如果已经 <10%，优先级降低。

这个和 §1.2 的动态 reward 同属"自适应课程"，可以组合：**用难度采样保证有信号、用动态 reward 引导学到什么**。

### 4.2 数据混合

GSM8K 偏简单（小学题），到后期模型容易"刷满"。把 `simple-reinforce++/rf++_vllm_one.py` 里的 MATH 数据按比例混进 GSM8K（比如 7:3），提升难度上限。注意 MATH 的 ground truth 是 `\boxed{}`，`reward_correct` 要兼容两种格式（`Auto_Program/hjy_grpo_program.py:190-205` 的 boxed 提取可直接借）。

---

## 5. 工具调用能力（迁移 Auto_Program 到主脚本）

`Auto_Program/` 已经实现了"模型写 Python → 执行 → 拿结果继续推理"的工具调用 CoT。你提到想把它纳入进阶实验。关键组件分布：

### 5.1 要搬的东西

| 组件 | 位置 | 作用 |
|---|---|---|
| system prompt | `Auto_Program/system_prompt_0312.txt` | 教模型用 `<program>` + ```python``` 块 |
| 代码执行 `run()` | `hjy_grpo_program.py:122-148` | `exec` 模型生成的 Python，1s 超时 |
| 多轮生成 `get_completions()` | `hjy_grpo_program.py:150-172` | 用 stop string `"The result of executing this Python code is:"` 截断 → 执行 → 拼回 → 递归续写 |
| `call_python` reward | `hjy_grpo_program.py:215-218` | `(代码块数 − 报错数) × 0.1` |
| 动态 reward 权重 | `hjy_grpo_program.py:238-242` | 早期重格式/工具，后期重正确率 |

把这几块搬进 `grpo_vllm_one.py` 的 `gen_worker` 即可（loss 完全不用动，GRPO_step 不变）。

### 5.2 ⚠️ 安全警告（必须知道）

`run()` 用 `exec(code, {}, local_vars)` **直接执行模型生成的代码**，唯一的保护是 `signal.alarm(1)` 超时。这意味着模型可以生成 `os.system("rm -rf ...")`、读写文件、发起网络请求。**在隔离机器/容器/无敏感数据的环境跑没问题**，但：
- 别在生产机、别挂载敏感目录、别有公网入口时裸跑；
- 想加固：用 `subprocess` 跑在最小权限容器里，或用一个白名单 `builtins`（屏蔽 `__import__`/`open`/`os`），只放 `print` + `math`/`numpy`。

### 5.3 `call_python` reward 的坑

`(python_cnt − error_cnt) × 0.1` 只奖励**"写了代码"**，会被 hack——模型可以刷无意义 ```python``` 块。改法：
- 只在**执行成功且产出非空**时给正分（`run()` 返回非 "Error!" 才算）；
- 或把工具 reward gate 在 format 对之后（先有结构再用工具）；
- 最严：只在"用了工具且最终 acc 对"时给，但那样信号太稀疏。

### 5.4 与 §1.2 动态 reward 的天然结合

工具调用引入了**第三个 reward 项**（correct / format / tool），三项之间更需要动态调度——这恰好是 `Auto_Program` 做的（按 step 切权重）。你可以把它升级成**按指标切**（correct_acc 上来就降 tool/format 权重），即 §1.2 方法在三维 reward 上的推广。这是一个有故事可讲的进阶实验。

---

## 6. 工程稳健性（让以上实验可信、可复现）

### 6.1 固定随机种子

当前**没有任何 seed**，A/B 实验的方差可能盖过改动效果。加：
```python
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
# vLLM 采样也有内部 rng，跨进程严格复现很难，但至少训练侧能定
```
每次实验跑 2-3 个 seed 报均值±方差，比单跑一次有说服力得多。

### 6.2 checkpoint resume

目前无断点续训。进阶实验跑得久（可能几千步），崩了重来代价大。建议：存 optimizer state（DeepSpeed 的 `engine.save_checkpoint`），启动时检测并 `engine.load_checkpoint`。注意 resume 要把 gen_worker 的 EMA / solve_rate（§1.2/§4.1）也一并存档，否则状态断层。

### 6.3 理解"步"与"迭代"的时序

训练侧每 `gen_update_steps=16` 步推一次 state_dict（`grpo_vllm_one.py:399`），gen_worker 每 `it % 3 == 0` 次迭代拉一次（`grpo_vllm_one.py:274`）——**这两边时间轴不同**（一个是训练 step，一个是生成 iteration）。所以"生成用的策略有多旧"是个动态值，调 `gen_update_steps` 时要同时看 ratio 分布（建议把 `ratio.mean()/max()` 也 log 进 swanlab，clip 频繁触发说明策略太旧）。

---

## 7. 推荐实验路线图

把上面的点排成一个由易到难、互相佐证的序列。每一步都配 §0 评测脚本 + §6.1 固定 seed。

| 顺序 | 实验 | 改动量 | 对照目的 |
|---|---|---|---|
| 0 | **补 eval_ckpt.py + 固定 seed + skip_rate/ratio 上报** | 小 | 后续一切的可比性地基 |
| 1 | baseline 复跑 + std eps 调大（§1.5）+ clip-higher（§3.1） | 小 | 拿到更稳的"加强 baseline" |
| 2 | **交换 correct/format 权重**（§1.1） | 极小 | 你计划的 A/B |
| 3 | **自适应动态 reward**（§1.2） | 中 | 你的旗舰想法 |
| 4 | Muon 混合优化器扫描（§2.2–2.6） | 大 | 你想做的优化器对照 |
| 5 | 难度采样（§4.1） | 中 | 降 skip_rate，提数据效率 |
| 6 | 熵奖励 / 双 advantage（§3.2 / §1.3） | 中大 | 治 reward hacking / 探索坍缩 |
| 7 | 迁移工具调用 + 三维动态 reward（§5） | 大 | 把 Auto_Program 融进来 |

**节奏建议**：0→1 是地基，务必先做；2→3→4 是你明确想做的三件事，每个都是独立 A/B，可并行开多卡跑；5 以后看前面结果再定。每个实验至少 2 个 seed，用 §0 脚本评 `eval/acc` / `eval/format_acc` 的曲线对比，别只看训练 loss。

---

## 8. 速查：每个实验"改哪几行 / 看什么指标"

| 实验 | 主要改动位置 | 关键观测指标 |
|---|---|---|
| 交换权重 | `grpo_vllm_one.py:28,32` | `eval/format_acc` 早期是否上得来、`eval/acc` 天花板 |
| 动态 reward | `gen_worker` 加 EMA + `gen_samples:248-254` 用动态 w_fmt | log `w_fmt`/`fmt_acc_ema`，看 `eval/acc` 转折 |
| Muon | `deepspeed.initialize` 传自定义 optimizer + 加 clip + 两组 lr 标定 | 前 50 步 loss 是否动、`eval/acc`、是否 NaN |
| clip-higher | `grpo_vllm_one.py:139` | ratio 分布、`eval/acc` 后期是否解封 |
| 熵奖励 | `GRPO_step` 加 entropy 项 | 输出多样性、`avg_completion_len` |
| 难度采样 | `gen_worker` 采样逻辑 | `batch/skip_rate`、单位时间有效梯度步数 |
| 工具调用 | 搬 `Auto_Program` 的 run/stop/call_python | `eval/acc`（尤其多步算术）、call_python hack 比例 |

> 写完任一项后，建议顺手把改动反映进 `build_run_tag()`（`grpo_vllm_one.py:70-83`）或手动给 `run_tag` 赋名，保证不同实验的 ckpt/swanlab 永不串台。
