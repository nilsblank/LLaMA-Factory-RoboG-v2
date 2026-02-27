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

# Install conda dependencies
conda install -c conda-forge libnpp-dev -y
conda install -c conda-forge cuda-nvrtc-dev -y

# Install torchcodec from wheels for CUDA 13
uv pip install torchcodec --extra-index-url https://download.pytorch.org/whl/cu130

echo "Environment setup complete!"