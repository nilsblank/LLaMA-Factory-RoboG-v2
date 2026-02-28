#!/bin/bash

#SBATCH -p booster
#SBATCH -A m3
#SBATCH -J RoboG
#SBATCH --cpus-per-task=64
#SBATCH --mem=0
##SBATCH --ntasks-per-node=1


# Cluster Settings
#SBATCH -t 12:00:00 ## 1-00:30:00 # 06:00:00 # 1-00:30:00 # 2-00:00:00


# Define the paths for storing output and error files
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


# -------------------------------
# Activate the virtualenv / conda environment

export HF_HOME=/e/home/jusers/blank4/jupiter/blank4/cache
export TRANSFORMERS_CACHE=$HF_HOME/transformers
export HF_DATASETS_CACHE=$HF_HOME/datasets
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

# Hard offline (prevents any network calls)
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"
export LD_LIBRARY_PATH=/e/home/jusers/blank4/jupiter/blank4/envs/miniforge3/envs/roboG/lib/python3.12/site-packages/nvidia/cu13/lib:$LD_LIBRARY_PATH
export WANDB_MODE="offline"


source ~/blank4/envs/miniforge3/etc/profile.d/conda.sh

conda activate roboG

export LD_LIBRARY_PATH=/home/hk-project-sustainebot/bm3844/miniconda3/envs/vlm/lib/python3.12/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
export TORCH_USE_CUDA_DSA=1
#export DISABLE_VERSION_CHECK=1

# NNODES=1
# NODE_RANK=0
# PORT=29500
# MASTER_ADDR=127.0.0.1
export CUDA_VISIBLE_DEVICES=0,1,2,3  

export PYTHONPATH=/e/home/jusers/blank4/jupiter/blank4/code/LLaMA-Factory-RoboG-v2/src:$PYTHONPATH
#srun llamafactory-cli train examples/train_full/qwen2vl_NILS_full_droid.yaml
#srun python src/llamafactory/cli.py train examples/train_full/qwen2_5vl_roboG_test.yaml
#srun python -m llamafactory.cli train examples/train_full/qwen2_5vl_roboG.yaml

if [[ $# -lt 1 ]]; then
  echo "Usage: sbatch launch.sh path/to/config.yaml [key=value ...]" >&2
  echo "Example: sbatch launch.sh config.yaml learning_rate=1e-5 num_train_epochs=3" >&2
  exit 1
fi

YAML_FILE="$1"
shift 

# YAML_FILE="$1"

# if [[ ! -f "$YAML_FILE" ]]; then
#   echo "YAML file not found: $YAML_FILE" >&2
#   exit 1
# fi

echo "Using config: $YAML_FILE"

# srun python -m llamafactory.cli train "$YAML_FILE"

OVERRIDE_ARGS=""
for arg in "$@"; do
  if [[ $arg == *"="* ]]; then
    OVERRIDE_ARGS="$OVERRIDE_ARGS --$arg"
  fi
done

if [[ -n "$OVERRIDE_ARGS" ]]; then
  echo "Applying overrides: $OVERRIDE_ARGS"
  srun python -m llamafactory.cli train "$YAML_FILE" $OVERRIDE_ARGS
else
  srun python -m llamafactory.cli train "$YAML_FILE"
fi


#run prediction with vllm

# srun python scripts/vllm_infer_from_cfg.py \
#     --config_path "$YAML_FILE" $OVERRIDE_ARGS


# #run evaluation

# srun python scripts/eval_boxes_poc_from_config.py \
#     --config_path "$YAML_FILE" $OVERRIDE_ARGS

#srun python -m llamafactory.cli train examples/train_full/qwen2_5vl_roboG_poc_box.yaml