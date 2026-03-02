#!/bin/bash
set -e
set -x

# Create and activate conda environment
conda create -n roboG python=3.12 -y

source ~/blank4/envs/miniforge3/etc/profile.d/conda.sh


conda activate roboG

# Install requirements
uv pip install -r requirements.txt

# Get and install lerobot in editable mode

uv pip install -e ../lerobot

#NOTE: Lerobot validates all existing episodes every time the dataset is loaded for each worker, which can be ver slow on cluster. To speed this up, we can disable the validation and assume that all existing episodes are valid. This is safe to do if we have already validated the dataset once and are not adding any new episodes. To disable validation, we can set the environment variable LEROBOT_SKIP_DATA_VALIDATION=1. This will skip the validation step and significantly speed up dataset loading on cluster.
#This requires changes in lerobot_dataset.py to check for this environment variable and skip validation if it is set. See the recent edits in lerobot_dataset.py for the implementation of this change.
export LEROBOT_SKIP_DATA_VALIDATION=1

# Install conda dependencies
conda install "ffmpeg" -c conda-forge
conda install -c conda-forge libnpp-dev -y
conda install -c conda-forge cuda-nvrtc-dev -y

# Install torchcodec from wheels for CUDA 13
uv pip install torchcodec --extra-index-url https://download.pytorch.org/whl/cu130


uv pip install flash_attn_3 --find-links https://windreamer.github.io/flash-attention3-wheels/cu130_torch2100


echo "Environment setup complete!"