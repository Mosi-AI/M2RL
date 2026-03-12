#!/bin/bash
mkdir -p raw
nvidia_datasets=(
    Nemotron-Math-Proofs-v1
    Nemotron-Math-v2
    Nemotron-Science-v1
    Nemotron-Competitive-Programming-v1
    Nemotron-Instruction-Following-Chat-v1
    Nemotron-Agentic-v1
)
# for dataset in "${nvidia_datasets[@]}"; do
#     hf download nvidia/$dataset --repo-type dataset --local-dir raw/$dataset
# done

datasets=(
    BAAI/TACO
    codeparrot/apps
    deepmind/code_contests
    open-r1/codeforces
)
for dataset in "${datasets[@]}"; do
    hf download $dataset --repo-type dataset --local-dir raw/$dataset
done