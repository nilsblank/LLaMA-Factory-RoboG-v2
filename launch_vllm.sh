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

# Check if model name is provided as argument
if [ -z "$1" ]; then
    echo "Error: No model name provided"
    echo "Usage: sbatch launch_vllm.sh <model_name>"
    echo "Example: sbatch launch_vllm.sh qwen3_5vl-8b"
    exit 1
fi

MODEL_NAME="$1"


DATASET=roboG_stagepoc_ablation_multi_frame_8_eval
SAVE_PATH="saves/${DATASET}/${MODEL_NAME}/generated_predictions.jsonl"

# Create the directory if it doesn't exist
mkdir -p "saves/${DATASET}/${MODEL_NAME}"

echo "Model name: $MODEL_NAME"
echo "Save path: $SAVE_PATH"

srun python scripts/vllm_infer.py \
    --model_name_or_path $MODEL_NAME \
    --template qwen3_vl_nothink \
    --dataset $DATASET \
    --enable_thinking False \
    --save_name "$SAVE_PATH" \
    --pipeline_parallel_size 1 \
    --image_max_pixels 65536 \
    --video_max_pixels 16384