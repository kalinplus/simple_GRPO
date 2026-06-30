# GRPO 变体说明

本项目包含 4 个 GRPO / RL 实现，共享同一个三进程架构（训练 + 生成 Worker + ref_server），但各有不同的算法侧重和工程取舍。

---

## 变体一览

| | **simple_grpo_v1/** | **simple-reinforce++/** | **Auto_Program/** |
|---|---|---|---|
| **算法** | GRPO | REINFORCE++ | GRPO |
| **生成方式** | PyTorch 原生 `.generate()` | vLLM | vLLM |
| **Loss** | GRPO（KL + clip） | REINFORCE++（无 KL，per-token advantage） | GRPO（KL + clip） |
| **归一化** | per-sequence mean → batch mean | 所有 token 求和 / num_items_in_batch | per-sequence mean → batch mean |
| **数据集** | GSM8K | MATH | 本地 JSON |
| **配置方式** | 硬编码 | `config.py` | `config.py` |
| **CoT 类型** | 纯文本 `💭...<answer>` | 纯文本 | 工具调用（Python 代码执行） |
| **日志** | print | print | wandb |

---

## 1. simple_grpo_v1/ — 早期纯 PyTorch 版本

**核心差异：不用 vLLM，用 PyTorch 原生 `.generate()` 做推理。**

### 要看的代码

- `grpo_ref_split.py:106-119` — `gen_samples()`：用 PyTorch 模型直接生成回答（不是 vLLM），返回 prompt_ids + output_ids 而非纯文本
- `grpo_ref_split.py:140-144` — gen_logps 计算：用同一个 PyTorch 模型前向算 logps（没有 vLLM 的 `prompt_logprobs` hack）
- `grpo_ref_split.py:168-169` — 可选 triton kernel `fast_log_softmax_gather`（默认注释掉），加速 per-token logps 计算
- `grpo_ref_split.py:151-154` — 支持 `genonly` 模式：`python grpo_ref_split.py genonly` 只跑生成不训练

### 与主版本（root）的区别

| | root | simple_grpo_v1 |
|---|---|---|
| 生成引擎 | vLLM（快、KV cache 管理好） | PyTorch `.generate()`（慢、无 KV cache 优化） |
| gen_logps | vLLM `prompt_logprobs` 单独 pass | 同一 PyTorch 模型前向计算 |
| batch 组织 | 全部在 gen_worker 内完成 | gen_mode 函数在训练循环内调用 |
| 序列化 | tensor + string 双协议 | tensor only |
| DeepSpeed batch | micro_batch=1, grad_accum=16 | micro_batch=8, grad_accum=2 |

### 预期效果

功能上与主版本等价（同样的 GRPO loss、同样的 reward），但**生成速度慢很多**，因为 PyTorch 的 `.generate()` 没有 vLLM 的 PagedAttention 和连续 batching 优化。适合理解最朴素的三进程拆分，不适合实际训练。

---

## 2. simple-reinforce++/ — REINFORCE++ 变体

**核心差异：去掉 KL 惩罚，改用 per-token advantage，loss 归一化方式不同。**

### 要看的代码

- `rf++_vllm_one.py:52-72` — **`REINFORCE_plusplus_step()`**：核心 loss 函数，和 GRPO 逐行对比就能看出差异
- `rf++_vllm_one.py:74-109` — gen_worker：和主版本结构相同，但数据集换成 MATH，group size = 1（`n=num_pre_Q` 改成了 `SamplingParams` 的 n 参数）
- `rf++_vllm_one.py:113-140` — reward 函数：`reward_correct` + `reward_format`，逻辑类似但 MATH 题的 ground truth 格式不同
- `config.py` — 统一配置文件，端口 51414（和主版本的 59875 不同，不能混用）

### Loss 对比（核心差异）

```
GRPO loss:
  per_token_loss = -min(ratio * advantage, clipped_ratio * advantage)
  loss = per_token_loss + beta * KL    ← 有 KL 惩罚
  loss = loss.mean_per_sequence.mean_per_batch  ← 两级均值

REINFORCE++ loss:
  per_token_loss = -min(ratio * per_token_advantage, clipped_ratio * per_token_advantage)
  loss = (per_token_loss * mask).sum() / num_items_in_batch  ← 单级归一化
  # 无 KL 惩罚项
```

三个关键区别：
1. **无 KL 惩罚** — 不需要 ref_logps 参与梯度计算（ref_server 仍负责传递数据，但不贡献 loss 项）
2. **per-token advantage** — advantage 是逐 token 的（gen_worker 里通过 reward 加权到每个 token），而非 GRPO 的 per-sequence scalar advantage
3. **归一化** — 所有有效 token 之和除以总 token 数，而非先 per-sequence 再 batch

### 预期效果

理论上更简洁（少一个超参 beta），但对 reward hacking 更敏感（没有 KL 锚点防止策略漂移）。使用 MATH 数据集（竞赛数学，比 GSM8K 难得多），训练步数更多（1200 步 vs 1000 步）。

---

## 3. Auto_Program/ — 工具调用 + 动态 reward

**核心差异：模型可以生成 Python 代码并执行验证，reward 权重随训练进度动态调整。**

### 要看的代码

- `hjy_grpo_program.py:67` — `GRPO_step()`：loss 本身和主版本相同，差异全在 reward 端
- `hjy_grpo_program.py:215-218` — **`call_python()`**：统计回答中 ````python` 代码块数量减去 `Error!` 出错次数，乘 0.1 作为额外 reward
- `hjy_grpo_program.py:228-242` — **动态 reward 权重**：训练前 16 步 `acc×1 + format×2 + call_python×2`（重格式），16 步后 `acc×2 + format×1 + call_python×1`（重准确率）
- `hjy_grpo_program.py:248-259` — 每轮生成数据写入 `gen_data_path` JSON 文件，用于离线分析
- `hjy_grpo_program.py:377-378` — wandb 日志初始化
- `config.py` — 包含 wandb_name / wandb_project / wandb_key / data_path 等配置
- `system_prompt_0312.txt` / `system_prompt_0312_zero.txt` — 系统提示词模板，引导模型使用 Python 工具

### 动态 reward 详解

```python
# 训练早期（step < 16×gen_update_steps）
reward = acc_score + 2*format_score + 2*call_python_score
# → 先让模型学会格式 + 工具调用

# 训练后期（step >= 16×gen_update_steps）
reward = 2*acc_score + format_score + call_python_score
# → 转为奖励准确率
```

### 预期效果

模型不仅能做文本推理，还能学会「写 Python 代码 → 执行 → 根据结果回答」这种 tool-use CoT 模式。适合需要精确计算的场景（如大数运算、方程求解）。wandb 日志 + 数据记录让训练过程更可追溯。

---

## 快速决策：该看哪个？

| 目的 | 看哪个 |
|---|---|
| 理解 GRPO 核心算法 | **root** (`grpo_vllm_one.py`) |
| 理解三进程架构的最朴素实现 | **simple_grpo_v1/** |
| 对比 GRPO 和 REINFORCE++ 的 loss 差异 | **simple-reinforce++/** 的 `REINFORCE_plusplus_step()` |
| 学习 reward 设计和动态权重 | **Auto_Program/** 的 `call_python()` + 动态 reward 部分 |
| 学习如何加 wandb / 工具调用能力 | **Auto_Program/** |
