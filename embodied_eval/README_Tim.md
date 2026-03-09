# Tim's Documentation

> [!NOTE]
> This README provides an overview of how to setup and use the benchmarks and models that I added.
> Since I had to install some packages, you can find my requirements file here: `embodied_eval/requirements_tim.txt`.
> However, you might have to adjust some entries since I installed them locally, such as this repo and `roboannotatorx`.

## Environment Setup

```bash
# Create env
conda create -n roboG python==3.11
conda activate roboG
pip install uv

# Dependecies listed in the project
cd ~/projects/LLaMA-Factory-RoboG-v2
uv pip install -r embodied_eval/requirements.txt
# If you require specific features, check the requirements folder for more dependencies

# Install project
uv pip install vllm==0.17  # LLAMA Factory installs 0.10.0, which causes problems with Qwen 3 VL
uv pip install hatchling editables  # Missing build packages
uv pip install -e . --no-build-isolation
uv pip install -r requirements/metrics.txt

# Dependency fixes
uv pip install hydra-core tensorflow-datasets qwen-vl-utils==0.0.14 torchmetrics pycocotools scikit-learn # sentence-transformers tf-keras
```

## Benchmarks

### V-STaR

**Installation**:

1. Download data with `git clone https://huggingface.co/datasets/V-STaR-Bench/V-STaR` (If there are problems try `git lfs install` before)
2. Unzip video folder
3. Make sure the `data_dir` in the config `benchmarks/vstar.yaml` points to the correct dataset folder

**Run benchmark**:

```bash
cd ~/projects/LLaMA-Factory-RoboG-v2/embodied_eval
conda activate roboG
python run_eval.py --config-path ./configs --config-name config_vstar
```

*Notes*:

- Make sure the amount of tokens is big enough, i.e. set `max_new_tokens: 16384` in `Qwen3VL_4B.yaml`
- Some questions have an unclear output format, e.g. when asking for timestamps it is not clear that only one start and end should be output

### Robo2VLM

**Installation**:

1. Make sure `HF_HOME` is set to a folder that does not have limited storage
2. `conda activate roboG`
3. Download dataset with `uvx --from huggingface_hub hf download keplerccc/ManipulationVQA --repo-type dataset --local-dir /hkfs/work/workspace/scratch/uemjv-roboG_datasets/Robo2VLM`
4. Make sure the `data_dir` in the config `benchmarks/robo2vlm.yaml` points to the correct dataset folder
5. Make sure you have an HF account with access to [LLaMA 3.2](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)

**Run benchmark**:

```bash
cd ~/projects/LLaMA-Factory-RoboG-v2/embodied_eval
conda activate roboG
hf auth login <user_name>
python run_eval.py --config-path ./configs --config-name config_robo2vlm
```

*Notes*:

- If there are problems with GPU memory, try adjusting the VLLM parameter `gpu_memory_utilization`, i.e. set `gpu_memory_utilization=0.6` in the model's config file and `gpu_memory_utilization=0.3` in `robo2vlm_benchmark.py`.

**Open Problems**:

- Using the example code loads the [larger dataset](https://huggingface.co/datasets/keplerccc/Robo2VLM-1) and not the [60k one](https://huggingface.co/datasets/keplerccc/ManipulationVQA-60k), which was probably used in the paper
- Category tag not in dataset (See [issue](https://huggingface.co/datasets/keplerccc/ManipulationVQA-60k/discussions/5))

## Models

### OpenAI

**Installation**:

```bash
export OPENAI_API_KEY=<api_key>  # Tip: Can be added to ~/.bashrc
uv pip install openai
```

**Run model**:

1. Set model in config file to `openai`
2. Run benchmark

### Gemini

**Info**:

- Uses file API in Google's `genai` interface for uploading larger videos, where videos are deleted after 48h ([Docs](https://ai.google.dev/gemini-api/docs/video-understanding))
- Should predict bounding boxes in coordinate range $[0,1000]$

**Installation**:

```bash
export GEMINI_API_KEY=<api_key>  # Tip: Can be added to ~/.bashrc
uv pip install google-genai
```

**Run model**:

1. Set model in config file to `gemini`
2. Run benchmark

### Open Router

**Info**:

- The available models that support video input are listed [here](https://openrouter.ai/models?fmt=cards&input_modalities=video)
- Videos can be passed inline as base64 ([Docs](https://openrouter.ai/docs/guides/overview/multimodal/videos))
- *Alternative*: Any-LLM does not seem to provide image and video input right now

>[!warning] Not implemented yet since only GPT and Gemini were needed

### RoboAnnotatorX

**Info**:

- No official instructions
- No stage 3 for 7b available, which is the one they use in their comparison table + benchmark script
- At the moment, it does not work due to a dependency conflict in the transformer version

**Installation**:

1. Create new environment like the default one, called `roboannotatorx`
2. Run:
	```bash
	cd <Your project folder>
	git clone https://github.com/TimWindecker/RoboannotatorX-RoboG.git
	cd RoboannotatorX-RoboG
	conda activate roboannotatorx
	uv pip install -e . --no-deps
	uv pip install timm==0.6.13 decord==0.6.0 transformers==4.35.2
	```

**Assemble model files**:

1. Create a new model folder (e.g. `/hkfs/work/workspace/scratch/uemjv-roboG_datasets/RoboAnnotatorX/roboannotatorx-13b-assembled`) and download all files from this table to it:

| File                      | Source                                                                                                                                                             |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `config.json`             | `wget https://huggingface.co/koulx/roboannotatorx-13b-stage3/resolve/main/roboannotatex-13b-4x-grid8-interval32-v3-stage3-31K-epoch-2/config.json`                 |
| `adapter_model.bin`       | `wget https://huggingface.co/koulx/roboannotatorx-13b-stage3/resolve/main/roboannotatex-13b-4x-grid8-interval32-v3-stage3-31K-epoch-2/adapter_model.bin`           |
| `adapter_config.json`     | `wget https://huggingface.co/koulx/roboannotatorx-13b-stage3/resolve/main/roboannotatex-13b-4x-grid8-interval32-v3-stage3-31K-epoch-2/adapter_config.json`         |
| `non_lora_trainables.bin` | `wget https://huggingface.co/koulx/roboannotatorx-13b-stage3/resolve/main/roboannotatex-13b-4x-grid8-interval32-v3-stage3-31K-epoch-2/non_lora_trainables.bin`     |
| `mm_projector.bin`        | `wget https://huggingface.co/koulx/roboannotator-13b-pretrain/resolve/main/roboannotatex-13b-4x-grid8-interval32-v3-pretrain-image-video-epoch-1/mm_projector.bin` |
| `special_tokens_map.json` | `wget https://huggingface.co/lmsys/vicuna-13b-v1.5/resolve/main/special_tokens_map.json`                                                                           |
| `tokenizer.model`         | `wget https://huggingface.co/lmsys/vicuna-13b-v1.5/resolve/main/tokenizer.model`                                                                                   |
| `tokenizer_config.json`   | `wget https://huggingface.co/lmsys/vicuna-13b-v1.5/resolve/main/tokenizer_config.json`                                                                             |

2. In `config.json` change line 15 from `"./llamavid/processor/clip-patch14-224"` to `"./roboannotatorx/processor/clip-patch14-224"`
3. In `config.json` change line 2 from `_name_or_path: ...` to `model_name_or_path: null`?
4. Download `wget https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/eva_vit_g.pth` to `./model_zoo/LAVIS/eva_vit_g.pth` in the folder with the model's code

### RynnBrain

**Info**:

- Based on Qwen 3 VL
- Uses normalized coordinates from 0-1000 for bboxes (See [bbox example](https://github.com/alibaba-damo-academy/RynnBrain/blob/main/cookbooks/4_object_location.ipynb))

**Installation**:

- Automatically downloads from HF
- As long as VLLM does not support transformers > `5.0` (see [Issue](https://github.com/vllm-project/vllm/pull/30566)), it is necessary to adjust the file `LLaMA-Factory-RoboG-v2/src/llamafactory/custom_models/qwen_3_vl_query_timechat.py` by removing the import for `maybe_autocast` and adding this after the imports (Make sure transformers is version `4.57.6` and vllm is `0.17.0`): 
	```
	def maybe_autocast(
	    device_type: str,
	    dtype: Optional["_dtype"] = None,
	    enabled: bool = True,
	    cache_enabled: bool | None = None,
	):
	    """
	    Context manager that only autocasts if:
	
	    - `autocast` is already enabled in this context
	    - Or this call to `maybe_autocast` has `enabled=True`
	
	    This prevents `autocast` being added to the graph when it is effectively a no-op.
	    Which makes graph splitting in `torch.compile` more flexible as it removes the
	    requirement that partition IDs be monotonically increasing.
	    """
	    if torch.is_autocast_enabled(device_type) or enabled:
	        return torch.autocast(device_type, dtype=dtype, enabled=enabled, cache_enabled=cache_enabled)
	    else:
	        return nullcontext()
	```

**Run model**:

1. Set model in config file to `rynnbrain_vllm` for fast evaluation, the config `rynnbrain` uses the HF model directly and is very slow
2. Run benchmark
