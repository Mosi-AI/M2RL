#!/bin/bash

set -ex

# will prevent ray from buffering stdout/stderr
export PYTHONBUFFERED=16

NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
N_GPUs=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -n 1)
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"
echo "N_GPUs: $N_GPUs"

sft_model="$M2RL_ROOT/checkpoints/Qwen3-4B-sft"
source "$M2RL_ROOT/slime/scripts/models/qwen3-4B.sh"

exp_name=Qwen3-4B-science-rl
CKPT_ARGS=(
   --hf-checkpoint Qwen/Qwen3-4B
   --ref-load $sft_model
   --load $M2RL_ROOT/checkpoints/Qwen3-4B-$exp_name
   --save $M2RL_ROOT/checkpoints/Qwen3-4B-$exp_name
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data $DATA_DIR/rl_train/science.parquet
   --input-key prompt
   --label-key label
   --apply-chat-template
   --rollout-shuffle
   --rm-type gpqa
   --num-rollout 400
   --rollout-batch-size 128
   --n-samples-per-prompt 16
   --rollout-max-response-len 32768
   --rollout-max-prompt-len 2048
   --rollout-temperature 1.
   --rollout-stop 151645

   # --over-sampling-batch-size 512
   # --dynamic-sampling-filter-path slime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std
   # --partial-rollout

   --global-batch-size 2048
   --balance-data
)

EVAL_ARGS=(
   --eval-interval 20
   --eval-prompt-data gpqa $DATA_DIR/val/gpqa.parquet
   --n-samples-per-eval-prompt 4
   --eval-max-response-len 32768
   --eval-top-p 0.7
)

PERF_ARGS=(
   --tensor-model-parallel-size 2
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 2
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 17600
)

GRPO_ARGS=(
   --advantage-estimator grpo
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
   --eps-clip 0.2
   --eps-clip-high 0.28

   --use-tis
   # --calculate-per-token-loss
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 2e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-tensorboard
   --tb-project-name qwen3-4B
   --tb-experiment-name $exp_name
   # --use-wandb
   # --wandb-project slime-dev
   # --wandb-group qwen3-4B-test
   # --wandb-key ${WANDB_KEY}
)

SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 2
   --sglang-mem-fraction-static 0.8

   --sglang-log-level warning
   --sglang-decode-log-interval 256
   --sglang-chunked-prefill-size 4096
)

MISC_ARGS=(
   # default dropout in megatron is 0.1
   --attention-dropout 0.0
   --hidden-dropout 0.0
   # should be good for model performance
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   # need to comment this when using model with MLA
   --attention-backend flash

   # --save-debug-rollout-data rollouts/{rollout_id}.txt
)

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"$MEGATRON_DIR\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"NCCL_DEBUG\": \"WARN\"
  }
}"

ray job submit --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes $WORLD_SIZE \
   --actor-num-gpus-per-node $N_GPUs \
   --colocate \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${ROLLOUT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${GRPO_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${SGLANG_ARGS[@]} \
   ${MISC_ARGS[@]}

