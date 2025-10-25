#!/bin/bash

#SBATCH -p accelerated
#SBATCH -A hk-project-p0024638
#SBATCH -J vllm_infer

# Cluster Settings
#SBATCH -c 64  # Number of cores per task
#SBATCH -t 4:00:00 ## 1-00:30:00 # 06:00:00 # 1-00:30:00 # 2-00:00:00
#SBATCH --gres=gpu:4


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
#srun llamafactory-cli train examples/train_full/qwen2vl_NILS_full_droid.yaml
#srun python src/llamafactory/cli.py train examples/train_full/qwen2_5vl_roboG_test.yaml
#srun python -m llamafactory.cli train examples/train_full/qwen2_5vl_roboG.yaml





srun python scripts/vllm_infer.py \
    --model_name_or_path /home/hk-project-sustainebot/bm3844/code/LLaMA-FactoryRoboG/saves/qwen2_5vl-3b/full/sft/roboG_stagepoc_ablation_two_frames_train \
    --template qwen2_vl \
    --dataset roboG_stagepoc_ablation_two_frames_train \
    --pipeline_parallel_size 4 \
    --image_max_pixels 262144 \
    --video_max_pixels 16384 \

#srun python -m llamafactory.cli train examples/train_full/qwen2_5vl_roboG_poc_box.yaml