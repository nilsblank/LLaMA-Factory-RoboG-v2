#!/bin/bash

#SBATCH -p accelerated
#SBATCH -A hk-project-p0024638
#SBATCH -J vllm_infer

# Cluster Settings
#SBATCH -c 64  # Number of cores per task
#SBATCH -t 4:00:00 ## 1-00:30:00 # 06:00:00 # 1-00:30:00 # 2-00:00:00
#SBATCH --gres=gpu:1


# Define the paths for storing output and error files
#SBATCH --output=/home/hk-project-sustainebot/bm3844/code/LLaMA-Factory-RoboG-v2/logs/outputs/%x_%j.out
#SBATCH --error=/home/hk-project-sustainebot/bm3844/code/LLaMA-Factory-RoboG-v2/logs/outputs/%x_%j.err


# -------------------------------
# Activate the virtualenv / conda environment


source ~/.bashrc
conda activate roboG_train

export LD_LIBRARY_PATH=/home/hk-project-sustainebot/bm3844/miniconda3/envs/vlm/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export TORCH_USE_CUDA_DSA=1
export DISABLE_VERSION_CHECK=1

# NNODES=1
# NODE_RANK=0
# PORT=29500
# MASTER_ADDR=127.0.0.1
#CUDA_VISIBLE_DEVICES=0,1,2,3  

export PYTHONPATH=/home/hk-project-sustainebot/bm3844/code/LLaMA-Factory-RoboG-v2/src:$PYTHONPATH

# Check if YAML config file is provided as argument
if [ -z "$1" ]; then
    echo "Error: No YAML config file provided"
    echo "Usage: sbatch launch_vllm.sh <path/to/config.yaml>"
    exit 1
fi

CONFIG_FILE="$1"

# Check if the config file exists
if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file '$CONFIG_FILE' not found"
    exit 1
fi

echo "Running vLLM inference with config: $CONFIG_FILE"

srun python scripts/vllm_infer_from_cfg.py \
    --config_path "$CONFIG_FILE"


#run evaluation
srun python scripts/eval_boxes_poc_from_config.py "$CONFIG_FILE" --save-images