#!/bin/bash
export HF_TOKEN=
mkdir -p val
python download_aime.py
python download_gpqa.py
python download_ifbench.py