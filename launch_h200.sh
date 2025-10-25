#!/bin/bash

#SBATCH -p accelerated-h200
#SBATCH -A hk-project-p0024638
#SBATCH -J RoboG

# Cluster Settings
#SBATCH -c 32  # Number of cores per task
#SBATCH -t 3:00:00 ## 1-00:30:00 # 06:00:00 # 1-00:30:00 # 2-00:00:00
#SBATCH --gres=gpu:1


# Define the paths for storing output and error files
#SBATCH --output=/home/hk-project-sustainebot/bm3844/code/LLaMA-FactoryRoboG/logs/outputs/%x_%j.out
#SBATCH --error=/home/hk-project-sustainebot/bm3844/code/LLaMA-FactoryRoboG/logs/outputs/%x_%j.err


# -------------------------------
# Activate the virtualenv / conda environment


source ~/.bashrc
conda activate roboG_train

export LD_LIBRARY_PATH=/home/hk-project-sustainebot/bm3844/miniconda3/envs/vlm/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export TORCH_USE_CUDA_DSA=1
# NNODES=1
# NODE_RANK=0
# PORT=29500
# MASTER_ADDR=127.0.0.1
#CUDA_VISIBLE_DEVICES=0,1,2,3  

export PYTHONPATH=/home/hk-project-sustainebot/bm3844/code/LLaMA-FactoryRoboG/src:$PYTHONPATH
#srun llamafactory-cli train examples/train_full/qwen2vl_NILS_full_droid.yaml
#srun python src/llamafactory/cli.py train examples/train_full/qwen2_5vl_roboG_test.yaml
#srun python -m llamafactory.cli train examples/train_full/qwen2_5vl_roboG.yaml

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch launch.sh path/to/config.yaml" >&2
  exit 1
fi

YAML_FILE="$1"

if [[ ! -f "$YAML_FILE" ]]; then
  echo "YAML file not found: $YAML_FILE" >&2
  exit 1
fi

echo "Using config: $YAML_FILE"

srun python -m llamafactory.cli train "$YAML_FILE"

#srun python -m llamafactory.cli train examples/train_full/qwen2_5vl_roboG_poc_box.yaml