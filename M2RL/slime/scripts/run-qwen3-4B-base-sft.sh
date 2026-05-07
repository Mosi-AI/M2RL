source "$M2RL_ROOT/slime/scripts/models/qwen3-4B.sh"

RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
NNODES=${NNODES:-1}

CKPT_ARGS=(
   --hf-checkpoint Qwen/Qwen3-4B
   --ref-load $M2RL_ROOT/checkpoints/Qwen3-4B-Base_torch_dist
   --loss-mask-type qwen3
   --load $M2RL_ROOT/checkpoints/Qwen3-4B-sft
   --save $M2RL_ROOT/checkpoints/Qwen3-4B-sft
   --save-interval 200
)

SFT_ARGS=(
   --rollout-function-path slime.rollout.sft_rollout.generate_rollout
   --prompt-data $DATA_DIR/sft/further_finetuning.jsonl
   --input-key messages
   --tool-key tools
   --rollout-shuffle
   --sft-rollout-workers 8
   --num-epoch 2
   --rollout-batch-size 512
   --global-batch-size 512

   --loss-type sft_loss
   --calculate-per-token-loss
   --disable-compute-advantages-and-returns
   --debug-train-only
)

PERF_ARGS=(
   --tensor-model-parallel-size 4
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
   --max-tokens-per-gpu 18432
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 5e-5
   --lr-decay-style cosine
   --min-lr 5e-6
   --lr-warmup-fraction 0.1
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.95
)

WANDB_ARGS=(
   --use-tensorboard
   --tb-project-name ./tensorboard_log
   --tb-experiment-name Qwen3-4B_SFT
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
)

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"$MEGATRON_DIR\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\",
    \"PYTORCH_CUDA_ALLOC_CONF\": \"expandable_segments:True\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train_async.py \
   --actor-num-nodes $NNODES \
   --actor-num-gpus-per-node 8 \
   ${MODEL_ARGS[@]} \
   ${CKPT_ARGS[@]} \
   ${SFT_ARGS[@]} \
   ${OPTIMIZER_ARGS[@]} \
   ${WANDB_ARGS[@]} \
   ${PERF_ARGS[@]} \
   ${EVAL_ARGS[@]} \
   ${MISC_ARGS[@]}
