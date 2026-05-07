#!/bin/bash

# for rerun the task
pkill -9 sglang
sleep 3
ray stop --force
pkill -9 ray
sleep 3
pkill -9 ray
set -ex

# 确保 Ray 能够找到本地的包目录
export PYTHONPATH=$PYTHONPATH:$(pwd)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export M2RL_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
export DATA_DIR=$M2RL_ROOT/data
export MEGATRON_DIR=$M2RL_ROOT/Megatron-LM

export SWANLAB_API_KEY="dYK4fySbCVN527Y3NtcVx"
ray stop --force

export MASTER_ADDR=${MASTER_ADDR:-"localhost"}
export RANK=${RANK:-0}
export WORLD_SIZE=${WORLD_SIZE:-1}

if [ $RANK -eq 0 ]; then
    ray start --head --node-ip-address=$MASTER_ADDR --port=6379 --num-gpus 8 --num-cpus=64 --include-dashboard=True
    echo "Starting Head Node at $MASTER_ADDR"
    start_time=$(date +%s)
    while true; do
        n_nodes=$(ray list nodes 2>/dev/null | grep "ALIVE" | wc -l)
        if [ "$n_nodes" -ge "$WORLD_SIZE" ]; then
            echo "All $WORLD_SIZE nodes have joined the Ray cluster."
            NNODES=$WORLD_SIZE bash $1
            break
        fi
        current_time=$(date +%s)
        elapsed_time=$((current_time - start_time))
        if [ $elapsed_time -gt 1200 ]; then
            echo "Timeout waiting for nodes to join the Ray cluster."
            exit 1
        fi
        sleep 5
    done
else
    echo "Worker node $RANK connecting to Head Node at $MASTER_ADDR"
    start_time=$(date +%s)
    while true; do
        if nc -z $MASTER_ADDR 6379; then
            echo "Ray Head Node is ready. Worker node $RANK connecting."
            ray start --address=$MASTER_ADDR:6379 --num-cpus=64
            echo "Worker node $RANK connected to Ray cluster."
            break
        else
            current_time=$(date +%s)
            elapsed_time=$((current_time - start_time))
            if [ $elapsed_time -gt 600 ]; then
                echo "Timeout waiting for Ray Head Node to be ready."
                exit 1
            fi
            sleep 5
        fi
    done

    while true; do
        if nc -z $MASTER_ADDR 6379; then
            sleep 1m
        else
            echo "Training Completed. Now exiting worker node $RANK."
            break
        fi
    done
fi
