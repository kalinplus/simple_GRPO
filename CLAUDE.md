# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A minimal GRPO (and REINFORCE++) implementation for RL-finetuning LLMs to reproduce R1-style `<think>/<answer>` reasoning. ~200 lines, no ray/accelerate/trl dependency — only `deepspeed`, `torch`, `vllm`, `transformers`, `bottle`. The GRPO loss formula is adapted from HuggingFace `trl`. Default task is GSM8K; reward = math correctness (`math_verify`) + answer-format compliance.

## Architecture — the three-process design

This is the key thing to understand; it spans every `.py` file. Training is split across three processes that never share Python state. They communicate over **HTTP (a `bottle` server) + a `torch.multiprocessing` Queue**, not NCCL:

1. **Training process** — `deepspeed <script>.py`, all training GPUs. Runs `GRPO_step`/`REINFORCE_plusplus_step`. Each step it **GETs** a batch from `ref_server /get` (blocks/waits if empty), does backward, and every `gen_update_steps` it pushes the latest `state_dict` into an `mp.Queue` to sync the generator.
2. **Generation worker** — spawned inside the training script via `mp.Process` (`gen_worker`), on **one dedicated GPU excluded from `CUDA_VISIBLE_DEVICES`** (`gen_device`). Runs vLLM, generates `num_pre_Q` answers/question, computes rewards + (optionally) `gen_logps` via a second vLLM pass (`prompt_logprobs`), and **POSTs** batches to `ref_server /upload`. Every few iterations it pulls a new `state_dict` off the Queue and hot-swaps vLLM weights via `llm_model.load_weights(...)`.
3. **Reference server** — `ref_server.py`, **one dedicated GPU** (`CUDA_VISIBLE_DEVICES=N python ref_server.py`). Holds the frozen ref model. Receives batches on `/upload`, computes per-token ref logps, puts the augmented batch on a `result_queue`, served out via `/get`.

**Data flow:** gen_worker →(HTTP POST `/upload`)→ ref_server computes ref logps → result_queue → training (HTTP GET `/get`) → GRPO loss → backward → (every `gen_update_steps`) state_dict → mp.Queue → gen_worker hot-swaps vLLM weights.

The ref model is decoupled onto its own GPU deliberately: it avoids torch multiprocessing duplicating the ref model across training ranks, and lets the ref/gen roles run on a *different machine* (e.g. cheap 4090s). That is also why `gen_device` must be left out of the training process's `CUDA_VISIBLE_DEVICES` and is remapped by `gen_worker` setting its own env.

### Serialization & the dual protocol
`ref_server.py` defines shared helpers (`tensor_to_bytes`/`bytes_to_tensor`/`make_bytes_list`/`bytes_list_to_list`) used by every script. The main script auto-switches between two upload formats based on the server's response: `'tensor'` (default — full token ids + logps, enables the PPO-ratio path) vs `'string'` (regroup using raw text). Don't change one side without the other.

### Loss paths
- `compute_gen_logps=True` (default in main): PPO-style clipped ratio `exp(policy - gen_logps)` against the *generation-time* logps. Requires the `gen_logps` vLLM pass.
- `compute_gen_logps=False`: falls back to `exp(policy - policy.detach()) * advantage` (no clipping, no ratio). See `GRPO_step`.
- REINFORCE++ (`simple-reinforce++/`) always uses the clipped ratio and normalizes by `num_items_in_batch` instead of GRPO's per-sequence mean.

Groups whose reward has near-zero variance (`max-min < 1e-4`) are skipped during generation — there is no advantage signal.

## Running

At least 2 GPUs (3+ recommended). Each variant is launched the same way: ref server on one terminal, training on another.

**Main GRPO** (root): params are **hardcoded at the top of `grpo_vllm_one.py`** (no argparse by design — "we need to try more possibilities than a fking long argparse"). Set `model_path`, `gen_device` (index relative to visible devices, must NOT be in the training `CUDA_VISIBLE_DEVICES`), `ref_server` URL.
```bash
# terminal 1 — ref model (own GPU)
CUDA_VISIBLE_DEVICES=7 python ref_server.py
# terminal 2 — training; gen_device=4 uses the GPU at that visible index
CUDA_VISIBLE_DEVICES=2,3,4,5,6 deepspeed grpo_vllm_one.py
```

**`simple-reinforce++/`** and **`Auto_Program/`**: params live in `config.py` (`base_config` / `train_config`), incl. `port`/`ref_server`/`train_gpu_num`. The ref server reads the same `config.py`, so the port matches automatically.
```bash
cd simple-reinforce++
CUDA_VISIBLE_DEVICES=0 python ref_server.py
CUDA_VISIBLE_DEVICES=1,2,3 deepspeed rf++_vllm_one.py
```

**`simple_grpo_v1/`**: earlier plain-split version with an optional triton loss kernel (`fast_log_softmax_gather`, commented out by default).

### Env / setup
- `pip install -r requirements.txt` pins `vllm==0.10.1.1`, deepspeed, torch, transformers, bottle, tornado, math-verify.
- The GSM8K example needs the `datasets` package (`pip install datasets`, possibly behind an HTTP proxy). **Without it the generator produces no data and the training loop sits forever on `waiting for batch...`** — `datasets` is intentionally not in requirements.
- Each variant's `ref_server` listens on a different port (59875 in root + `simple_grpo_v1`, configurable in the other two). They are not interchangeable — run the `ref_server.py` from the same directory as the training script.

## The four variants

| Dir | Algorithm | Config | Extras |
|---|---|---|---|
| root (`grpo_vllm_one.py`) | GRPO | hardcoded at top | vLLM gen, dual tensor/string protocol |
| `simple_grpo_v1/` | GRPO | hardcoded | optional triton loss kernel |
| `simple-reinforce++/` | REINFORCE++ | `config.py` | `num_items_in_batch` norm, multi-GPU train |
| `Auto_Program/` | GRPO | `config.py` | tool-use/program-calling CoT, wandb logging, richer reward shaping (acc/format/`call_python`), system prompts as `.txt` files |

`Auto_Program` sets `NCCL_P2P_DISABLE=1` and logs to wandb + records generated data to `record_path`/`gen_data_path`; it loads training data from a local JSON (`data_path`), not GSM8K.
