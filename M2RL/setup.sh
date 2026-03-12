#!/bin/bash

set -ex

apt-get update && apt-get install -y libnuma-dev
# Add micromamba to PATH
export PATH="/root/.local/bin:$PATH"

yes '' | "${SHELL}" <(curl -L micro.mamba.pm/install.sh)
export PS1=tmp
mkdir -p /root/.cargo/
touch /root/.cargo/env
source ~/.bashrc

micromamba create -n slime python=3.12 pip -c conda-forge -y

# Initialize micromamba shell integration for activation to work
eval "$(micromamba shell hook --shell=bash)"

micromamba activate slime
export CUDA_HOME="$CONDA_PREFIX"
export MEGATRON_COMMIT="29eed5dcbc753f6171d48ba60095ef392108e74b"

# install cuda 12.9 as it's the default cuda version for torch
micromamba install -n slime cuda cuda-nvtx cuda-nvtx-dev nccl -c nvidia/label/cuda-12.9.1 -y
micromamba install -n slime -c conda-forge cudnn -y

pip install cuda-python==13.1.0
pip install torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 --index-url https://download.pytorch.org/whl/cu129

pip install sglang==0.5.6
pip install cmake ninja

# flash attn
# the newest version megatron supports is v2.7.4.post1
MAX_JOBS=64 pip -v install flash-attn==2.7.4.post1 --no-build-isolation

pip install git+https://github.com/ISEEKYAN/mbridge.git@89eb10887887bc74853f89a4de258c0702932a1c --no-deps
pip install --no-build-isolation "transformer_engine[pytorch]==2.10.0"
pip install flash-linear-attention==0.4.0
NVCC_APPEND_FLAGS="--threads 4" \
  pip -v install --disable-pip-version-check --no-cache-dir \
  --no-build-isolation \
  --config-settings "--build-option=--cpp_ext --cuda_ext --parallel 8" git+https://github.com/NVIDIA/apex.git@10417aceddd7d5d05d7cbf7b0fc2daad1105f8b4

pip install git+https://github.com/fzyzcjy/torch_memory_saver.git@dc6876905830430b5054325fa4211ff302169c6b --no-cache-dir --force-reinstall
pip install git+https://github.com/fzyzcjy/Megatron-Bridge.git@dev_rl --no-build-isolation
pip install "nvidia-modelopt[torch]>=0.37.0" --no-build-isolation

# Megatron
# git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM
# git checkout $MEGATRON_COMMIT
pip install .

# Slime
cd ../slime
pip install -e .

pip install polars datasets==3.6.0 dill==0.3.7
pip install pebble latex2sympy2 word2number
pip install nltk spacy emoji syllapy langdetect immutabledict
