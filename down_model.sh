#!/bin/bash

# pip install -U "huggingface_hub[cli]"
export HF_ENDPOINT=https://hf-mirror.com

# Qwen2.5 base models for GRPO / REINFORCE++ training
# model_path in grpo_vllm_one.py points at ./models/Qwen/Qwen2.5-3B
MODEL_ROOT=/mnt/workspace/hkl/simple_GRPO/models/Qwen

hf download Qwen/Qwen2.5-3B --local-dir "${MODEL_ROOT}/Qwen2.5-3B"
hf download Qwen/Qwen2.5-7B --local-dir "${MODEL_ROOT}/Qwen2.5-7B"
