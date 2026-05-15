#!/bin/bash
TEACHER_IP="109.22.106.170"
TEACHER_PORT_math=1234
TEACHER_PORT_code=2345
TEACHER_PORT_science=3456
TEACHER_PORT_if=4567
TEACHER_IP_agent="109.22.106.170"
TEACHER_PORT_agent=5678

# Wait for the teacher model server to be ready
until curl -sf http://$TEACHER_IP:$TEACHER_PORT_math/health_generate > /dev/null; do
    echo "Waiting for the math teacher model server to start..."
    sleep 2
done
echo "Math teacher model server is up and running at $TEACHER_IP:$TEACHER_PORT_math."

until curl -sf http://$TEACHER_IP:$TEACHER_PORT_code/health_generate > /dev/null; do
    echo "Waiting for the code teacher model server to start..."
    sleep 2
done
echo "Code teacher model server is up and running at $TEACHER_IP:$TEACHER_PORT_code."

until curl -sf http://$TEACHER_IP:$TEACHER_PORT_science/health_generate > /dev/null; do
    echo "Waiting for the science teacher model server to start..."
    sleep 2
done
echo "Science teacher model server is up and running at $TEACHER_IP:$TEACHER_PORT_science."

until curl -sf http://$TEACHER_IP:$TEACHER_PORT_if/health_generate > /dev/null; do
    echo "Waiting for the if teacher model server to start..."
    sleep 2
done
echo "IF teacher model server is up and running at $TEACHER_IP:$TEACHER_PORT_if."
sleep 10

until curl -sf http://$TEACHER_IP_agent:$TEACHER_PORT_agent/health_generate > /dev/null; do
    echo "Waiting for the agent teacher model server to start..."
    sleep 2
done
echo "Agent teacher model server is up and running at $TEACHER_IP_agent:$TEACHER_PORT_agent."
sleep 10

# student config
NVLINK_COUNT=$(nvidia-smi topo -m 2>/dev/null | grep -o 'NV[0-9][0-9]*' | wc -l)
if [ "$NVLINK_COUNT" -gt 0 ]; then
    HAS_NVLINK=1
else
    HAS_NVLINK=0
fi
echo "HAS_NVLINK: $HAS_NVLINK (detected $NVLINK_COUNT NVLink references)"

source "scripts/models/qwen3-4B.sh"

RAY_ADDRESS=${RAY_ADDRESS:-"http://localhost:8265"}
NNODES=${NNODES:-1}

CKPT_ARGS=(
   --hf-checkpoint /cpfs01/haoqingwang/slime_output/Qwen3-4B_SFT_Further/iter_0000999_hf
   --ref-load /cpfs01/haoqingwang/slime_output/Qwen3-4B_SFT_Further
   --load /cpfs01/liziheng/slime_output/MT_OPD_Qwen3_4B_TEST
   --save /cpfs01/liziheng/slime_output/MT_OPD_Qwen3_4B_TEST
   --save-interval 40
)

ROLLOUT_ARGS=(
   --prompt-data /cpfs01/liziheng/datasets/Nemotron-3-Nano-RL-Training-Blend/train2.parquet
   --input-key prompt
   --apply-chat-template
   --rollout-shuffle
   --num-rollout 400
   --rollout-batch-size 256
   --n-samples-per-prompt 4
   --global-batch-size 1024
   --rollout-max-response-len 16384
   --rollout-temperature 1.
   --rollout-max-prompt-len 2048
   --rollout-stop 151645 151646

   --balance-data
)

RM_ARGS=(
   --custom-rm-path on_policy_distillation.reward_func
   --custom-reward-post-process-path on_policy_distillation.post_process_rewards
   --rm-url-math http://$TEACHER_IP:$TEACHER_PORT_math/generate
   --rm-url-if http://$TEACHER_IP:$TEACHER_PORT_if/generate
   --rm-url-code http://$TEACHER_IP:$TEACHER_PORT_code/generate
   --rm-url-science http://$TEACHER_IP:$TEACHER_PORT_science/generate
   --rm-url-agent http://$TEACHER_IP_agent:$TEACHER_PORT_agent/generate
)

EVAL_ARGS=(
)

PERF_ARGS=(
   --tensor-model-parallel-size 8
   --sequence-parallel
   --pipeline-model-parallel-size 1
   --context-parallel-size 1
   --expert-model-parallel-size 1
   --expert-tensor-parallel-size 1

   --recompute-granularity full
   --recompute-method uniform
   --recompute-num-layers 1

   # --micro-batch-size 1
   --use-dynamic-batch-size
   --max-tokens-per-gpu 18432
)

GRPO_ARGS=(
   --advantage-estimator on_policy_distillation
   --use-kl-loss
   --kl-loss-coef 0.00
   --kl-loss-type low_var_kl
   --entropy-coef 0.00
)

OPTIMIZER_ARGS=(
   --optimizer adam
   --lr 1e-6
   --lr-decay-style constant
   --weight-decay 0.1
   --adam-beta1 0.9
   --adam-beta2 0.98
)

WANDB_ARGS=(
   --use-tensorboard
   --tb-project-name ./tensorboard_log
   --tb-experiment-name MT_OPD_Qwen3_4B_RL400
)

# student sglang
SGLANG_ARGS=(
   --rollout-num-gpus-per-engine 1
   --sglang-mem-fraction-static 0.6
)

MISC_ARGS=(
   --attention-dropout 0.0
   --hidden-dropout 0.0
   --accumulate-allreduce-grads-in-fp32
   --attention-softmax-in-fp32
   --attention-backend flash
)

CUSTOM_ARGS=(
   --custom-generate-function-path slime.rollout.custom_rollout.generate
)

# Build the runtime environment JSON with proper variable substitution
RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/cpfs01/haoqingwang/Megatron-LM/\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"NCCL_NVLS_ENABLE\": \"${HAS_NVLINK}\"
  }
}"

ray job submit --address="http://127.0.0.1:8265" \
   --runtime-env-json="${RUNTIME_ENV_JSON}" \
   -- python3 train.py \
   --actor-num-nodes $NNODES \
   --actor-num-gpus-per-node 8 \
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
   ${MISC_ARGS[@]} \
   ${RM_ARGS[@]} \
   ${CUSTOM_ARGS[@]}
