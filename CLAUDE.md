# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is **LLaMA Factory RoboG** — a fork of [LLaMA Factory](https://github.com/hiyouga/LLaMA-Factory) extended for robotics/embodied-AI fine-tuning. The core additions are:
- Custom Qwen2.5-VL and Qwen3-VL model variants with temporal resamplers (`src/llamafactory/custom_models/`)
- LeRobot dataset integration via `lerobot://` URI scheme (`src/llamafactory/data/lerobot_bridge.py`)
- Lance columnar format dataset support (`src/llamafactory/data/lance_utils.py`)
- WebDataset (TAR shard) streaming support in the data loader
- Embodied-AI evaluation framework (`embodied_eval/`)
- SLURM launch scripts for the HoreKa/LUMI clusters

The `roboG/` directory is a Python virtualenv — ignore it.

## Commands

```bash
# Lint and format
make style       # auto-fix with ruff
make quality     # check only

# Run tests
make test        # WANDB_DISABLED=true pytest -vv --import-mode=importlib tests/ tests_v1/

# Single test file
WANDB_DISABLED=true pytest -vv --import-mode=importlib tests/test_data_utils.py

# License headers check
make license

# Build package
make build

# Run training (local)
PYTHONPATH=src python -m llamafactory.cli train examples/train_full/qwen2_5vl_roboG_poc_box.yaml

# Run training with overrides
PYTHONPATH=src python -m llamafactory.cli train examples/train_full/qwen2_5vl_roboG_poc_box.yaml learning_rate=1e-5

# Launch on SLURM cluster (passes YAML + optional key=value overrides)
sbatch launch.sh examples/train_full/qwen2_5vl_roboG_poc_box.yaml

# vLLM inference from a training config
python scripts/vllm_infer_from_cfg.py --config_path examples/train_full/qwen2_5vl_roboG_poc_box.yaml

# Evaluate bounding-box predictions
python scripts/eval_boxes_poc_from_config.py --config_path examples/train_full/qwen2_5vl_roboG_poc_box.yaml

# Prepare LeRobot VQA dataset as WebDataset
python scripts/prepare_lerobot_vqa.py --lerobot-dataset /path/to/lerobot --export-webdataset --webdataset-dir /out
```

## Architecture

### v0 vs v1

The CLI dispatches based on `USE_V1` env var (`src/llamafactory/cli.py`):
- **v0 (default)**: `src/llamafactory/` — dependency order: `api`/`webui` → `chat`/`eval`/`train` → `data`/`model` → `hparams` → `extras`
- **v1** (`USE_V1=1`): `src/llamafactory/v1/` — `trainers` → `core` → `accelerator`/`plugins`/`config` → `utils`

### Core v0 modules

| Module | Purpose |
|---|---|
| `hparams/` | Dataclass definitions for all CLI arguments |
| `model/loader.py` + `model/patcher.py` | Load and patch HF models (LoRA, quantization, attention patches) |
| `data/loader.py` | Dataset loading; reads `data/dataset_info.json` for dataset registry |
| `data/template.py` | Chat templates per model family |
| `data/mm_plugin.py` | Multimodal preprocessing plugin (image/video decoding, vision token injection) |
| `data/lerobot_bridge.py` | Lazy LeRobot frame loading via `lerobot://` URIs resolved at collation time |
| `data/lance_utils.py` | Lance format iterator (supports blob columns for video) |
| `train/sft/`, `train/dpo/`, `train/ppo/`, `train/rm/`, `train/kto/`, `train/pt/` | Per-method trainer + workflow |
| `train/mca/` | RoboG-specific multi-camera/action trainer |

### Custom models (RoboG additions)

`src/llamafactory/custom_models/` contains:
- `modeling_qwen2_5_vl_roboG.py` / `modeling_qwen3_vl_roboG.py` — patched VL model classes
- `temporal_resamplers/` — slot-attention-based temporal resampler (HybridSlotQuery), text-conditioned VLM resampler; these add a temporal abstraction stage between the vision encoder and the LLM

### Dataset formats

All datasets are registered in `data/dataset_info.json`. Supported source formats:
- Standard JSON/JSONL (alpaca or sharegpt `formatting`)
- **WebDataset**: `"webdataset_files": "shards-{0000..0127}.tar"` — each TAR sample has `sample.json` + `sample.mp4`/`sample.jpg`
- **LeRobot**: `"lerobot_files": "path/to/vqa.jsonl"` with `lerobot://` image references resolved at collation
- **Lance**: `"lance_files": "path/to/data.lance"` — supports blob v2 columns for video

Always use `streaming: true` in training configs when using IterableDataset sources (WebDataset, LeRobot, Lance).

### Training config structure

YAML configs in `examples/train_full/` specify model, method, dataset, output, and training hyperparameters. Key RoboG-specific fields:
```yaml
freeze_vision_tower: true
freeze_multi_modal_projector: true
evaluators:
  - bbox_evaluator
```

## Code Style

- Linter/formatter: **ruff** (line length 119, Google-style docstrings)
- All source files must carry the Apache 2.0 license header (`make license` checks)
- 2 blank lines after imports; double quotes
