# WebDataset Export for LeRobot Episodes

## Overview

The `prepare_lerobot_vqa.py` script now supports exporting LeRobot episodes as WebDataset tar files with actual video bytes. This is useful for:

- **Efficient storage**: Videos are compressed as MP4 files
- **Fast streaming**: WebDataset format enables efficient sequential reading
- **Multi-turn conversations**: Support for multiple Q&A pairs per video
- **Multiprocessing**: Parallel extraction for faster processing
- **VQA integration**: Merge existing VQA datasets with episode videos

## Output Format

Each sample in the WebDataset contains:

```python
{
    "__key__": "bridge_episode_000042_observation_images_camera_0",  # Unique identifier
    "mp4": <video_bytes>,  # Actual MP4 video bytes
    "json": {
        "conversations": [
            [  # Conversation 1
                {"role": "user", "content": "<video>\nWhat is the robot doing?"},
                {"role": "assistant", "content": "Picking up a cup"}
            ],
            [  # Conversation 2
                {"role": "user", "content": "<video>\nWhat color is the object?"},
                {"role": "assistant", "content": "Blue"}
            ]
        ],
        "episode_idx": 42,
        "camera_key": "observation.images.camera_0",
        "dataset": "bridge"
    }
}
```

## Installation

Ensure you have the required dependencies:

```bash
pip install webdataset opencv-python
```

## Basic Usage

### 1. Export Single Dataset with Default Questions

```bash
python scripts/prepare_lerobot_vqa.py \
    --lerobot-dataset /path/to/lerobot/dataset \
    --camera-key observation.images.front \
    --enumerate \
    --as-video \
    --export-webdataset \
    --webdataset-dir /output/webdataset \
    --question-template "What task did the robot perform in this video?"
```

This creates tar files where:
- Each episode becomes one video sample
- Questions use the template, answers use the episode's language instruction
- Videos are encoded at 10 fps by default

### 2. Export with VQA Dataset (Multiple Q&A Pairs)

If you have an existing VQA dataset with Q&A pairs per episode:

```bash
python scripts/prepare_lerobot_vqa.py \
    --lerobot-dataset /path/to/lerobot/dataset \
    --camera-key observation.images.front \
    --enumerate \
    --as-video \
    --export-webdataset \
    --vqa-dataset /path/to/vqa_questions.jsonl \
    --webdataset-dir /output/webdataset_with_vqa
```

**VQA dataset format** (JSONL):
```json
{"episode_id": 42, "messages": [{"role": "user", "content": "What is the robot doing?"}, {"role": "assistant", "content": "Picking up a cup"}]}
{"episode_id": 42, "messages": [{"role": "user", "content": "What color is the object?"}, {"role": "assistant", "content": "Blue"}]}
{"episode_id": 43, "messages": [{"role": "user", "content": "Where is the robot?"}, {"role": "assistant", "content": "In a kitchen"}]}
```

All Q&A pairs with the same `episode_id` are grouped into the `conversations` array for that video.

### 3. Multi-Dataset Export

Export multiple datasets with different configurations:

```bash
python scripts/prepare_lerobot_vqa.py \
    --lerobot-datasets '{"bridge": "/data/bridge", "droid": "/data/droid"}' \
    --camera-keys '{"bridge": "observation.images.camera_0", "droid": "wrist_camera"}' \
    --enumerate \
    --as-video \
    --export-webdataset \
    --webdataset-dir /output/multi_dataset \
    --question-template "What task did the robot perform?"
```

This creates separate subdirectories:
- `/output/multi_dataset/bridge/bridge_shard-000000.tar`
- `/output/multi_dataset/droid/droid_shard-000000.tar`

### 4. Diverse Language Subsampling

Ensure balanced representation of different tasks:

```bash
python scripts/prepare_lerobot_vqa.py \
    --lerobot-dataset /path/to/lerobot/dataset \
    --camera-key observation.images.front \
    --enumerate \
    --as-video \
    --export-webdataset \
    --subsample-diverse-language \
    --webdataset-dir /output/diverse
```

This limits episodes per unique language instruction (default: 50 per instruction).

## Advanced Options

### Video Settings

- `--video-fps 15`: Set video frame rate (default: 10)
- `--samples-per-shard 50`: Samples per tar file (default: 100)

### Camera Settings

- `--camera-key-mode all`: Export all cameras (one sample per camera)
- `--camera-key-mode sample`: Random camera selection (one sample per episode)

### Performance

- `--num-workers 16`: Parallel processing workers (default: 16)

### Episode Selection

- `--episodes 0 1 2 5 10`: Export only specific episodes

## Reading WebDataset Output

### Python Example

```python
import webdataset as wds
import json

dataset = wds.WebDataset("/output/webdataset/*.tar")

for sample in dataset:
    key = sample["__key__"]
    video_bytes = sample["mp4"]
    metadata = json.loads(sample["json"])
    
    print(f"Episode {metadata['episode_idx']}:")
    for i, conv in enumerate(metadata["conversations"]):
        print(f"  Q{i+1}: {conv[0]['content']}")
        print(f"  A{i+1}: {conv[1]['content']}")
    
    # Save video to file
    with open(f"{key}.mp4", "wb") as f:
        f.write(video_bytes)
```

### Inspect with Provided Script

```bash
python scripts/read_webdataset_example.py "/output/webdataset/*.tar"
```

## Complete Workflow Example

```bash
# 1. Create VQA dataset from LeRobot episodes
python scripts/prepare_lerobot_vqa.py \
    --lerobot-dataset /data/bridge_v2 \
    --enumerate \
    --as-video \
    --output /data/processed/bridge_episodes.jsonl

# 2. Generate additional Q&A pairs (your custom script)
python your_vqa_generation_script.py \
    --input /data/processed/bridge_episodes.jsonl \
    --output /data/processed/bridge_vqa_enriched.jsonl

# 3. Export to WebDataset with enriched Q&A
python scripts/prepare_lerobot_vqa.py \
    --lerobot-dataset /data/bridge_v2 \
    --enumerate \
    --as-video \
    --export-webdataset \
    --vqa-dataset /data/processed/bridge_vqa_enriched.jsonl \
    --webdataset-dir /data/webdataset/bridge \
    --samples-per-shard 100 \
    --video-fps 15 \
    --num-workers 16
```

## Performance Tips

1. **Multiprocessing**: Use `--num-workers` based on CPU cores (default: 16)
2. **Shard size**: Larger shards (200-500 samples) for better I/O efficiency
3. **Video FPS**: Lower FPS (5-10) reduces file size without losing much information
4. **Diverse sampling**: Use `--subsample-diverse-language` to prevent task imbalance
5. **Storage**: Plan for ~1-5MB per video sample (depends on resolution and duration)

## Troubleshooting

### "webdataset not available"
```bash
pip install webdataset opencv-python
```

### "No frames extracted for episode X"
- Check camera key availability
- Verify episode index is valid
- Try different camera key

### Out of memory
- Reduce `--num-workers`
- Process in batches using `--episodes`
- Use `--samples-per-shard` to create smaller tar files

### Slow processing
- Increase `--num-workers` (up to CPU count)
- Use SSD storage for LeRobot dataset
- Consider `--camera-key-mode sample` instead of `all`

## Integration with Training

After creating WebDataset:

1. **Update dataset_info.json**:
```json
{
    "bridge_webdataset": {
        "file_name": "/data/webdataset/bridge/*.tar",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "videos": "videos"
        }
    }
}
```

2. **Load in training script**:
```python
import webdataset as wds

dataset = (
    wds.WebDataset("/data/webdataset/bridge/*.tar")
    .decode()
    .to_tuple("__key__", "mp4", "json")
)
```

## Notes

- Videos are saved only once per episode-camera combination
- Multiprocessing ensures efficient parallel extraction
- WebDataset format enables streaming from disk/cloud storage
- Compatible with PyTorch DataLoader for training
