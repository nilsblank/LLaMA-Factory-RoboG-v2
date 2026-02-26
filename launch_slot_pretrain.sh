#!/bin/bash
#SBATCH -A hk-project-pai00093
#SBATCH -p accelerated-h200
#SBATCH --time=10:00:00
#SBATCH --gres=gpu:1
#SBATCH --job-name=slot_pretrain

#SBATCH --output=/home/hk-project-sustainebot/bm3844/code/LLaMA-Factory-RoboG-v2/logs/outputs/%x_%j.out
#SBATCH --error=/home/hk-project-sustainebot/bm3844/code/LLaMA-Factory-RoboG-v2/logs/outputs/%x_%j.err



source /home/hk-project-sustainebot/bm3844/miniconda3/etc/profile.d/conda.sh

conda activate robog

export LD_LIBRARY_PATH=/home/hk-project-sustainebot/bm3844/miniconda3/envs/robog/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export TORCH_USE_CUDA_DSA=1
export DISABLE_VERSION_CHECK=1
export PYTHONPATH=/home/hk-project-sustainebot/bm3844/code/LLaMA-Factory-RoboG-v2/src:$PYTHONPATH


python -m llamafactory.cli train \
  /home/hk-project-sustainebot/bm3844/code/LLaMA-Factory-RoboG-v2/examples/train_full/qwen3vl/qwen3vl_roboG_poc_box_qwen_reasoning_debug.yaml