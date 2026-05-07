#!/bin/bash

# Convert Qwen3-4B-Base from HuggingFace format to Megatron torch_dist format

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export M2RL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export MEGATRON_DIR=$M2RL_ROOT/Megatron-LM

cd $SCRIPT_DIR/..

# Load model architecture arguments
source scripts/models/qwen3-4B.sh

# HF model path (HuggingFace remote repo)
HF_CHECKPOINT="Qwen/Qwen3-4B"

# Output path (relative to slime directory)
OUTPUT_DIR="$M2RL_ROOT/checkpoints/Qwen3-4B-Base_torch_dist"

# Convert (single GPU)
PYTHONPATH=$MEGATRON_DIR python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint ${HF_CHECKPOINT} \
    --save ${OUTPUT_DIR}