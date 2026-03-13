#!/bin/bash
mkdir -p raw
mkdir -p train
hf download nvidia/Nemotron-3-Nano-RL-Training-Blend --repo-type dataset --local-dir raw/Nemotron-3-Nano-RL-Training-Blend
python create_nanov3_jsonl.py --input raw/train.jsonl --output raw/train_complete.jsonl
python process_train.py
