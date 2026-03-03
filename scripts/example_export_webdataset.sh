#!/bin/bash
# Example: Export LeRobot episodes to webdataset format with actual video bytes

# Example 1: Export single dataset with default questions
# This will create tar files with videos and language instructions from episodes
python scripts/prepare_lerobot_vqa.py \
    --lerobot-dataset /path/to/lerobot/dataset \
    --camera-key observation.images.front \
    --enumerate \
    --as-video \
    --export-webdataset \
    --webdataset-dir /output/webdataset \
    --question-template "What task did the robot perform in this video?" \
    --samples-per-shard 100 \
    --video-fps 10 \
    --num-workers 8

# Example 2: Export with VQA dataset (merge Q&A pairs)
# If you have a VQA dataset with episode_id and Q&A pairs, merge them
python scripts/prepare_lerobot_vqa.py \
    --lerobot-dataset /path/to/lerobot/dataset \
    --camera-key observation.images.front \
    --enumerate \
    --as-video \
    --export-webdataset \
    --vqa-dataset /path/to/vqa_dataset.jsonl \
    --webdataset-dir /output/webdataset_with_vqa \
    --samples-per-shard 100 \
    --video-fps 10 \
    --num-workers 8

# Example 3: Multi-dataset export with diverse language subsampling
python scripts/prepare_lerobot_vqa.py \
    --lerobot-datasets '{"bridge": "/data/bridge", "droid": "/data/droid"}' \
    --camera-keys '{"bridge": "observation.images.camera_0", "droid": "wrist_camera"}' \
    --enumerate \
    --as-video \
    --export-webdataset \
    --webdataset-dir /output/multi_dataset \
    --question-template "What task did the robot perform?" \
    --subsample-diverse-language \
    --camera-key-mode sample \
    --samples-per-shard 50 \
    --video-fps 10 \
    --num-workers 16

# Example 4: Export specific episodes only
python scripts/prepare_lerobot_vqa.py \
    --lerobot-dataset /path/to/lerobot/dataset \
    --camera-key observation.images.front \
    --enumerate \
    --episodes 0 1 2 5 10 \
    --as-video \
    --export-webdataset \
    --webdataset-dir /output/specific_episodes \
    --samples-per-shard 10 \
    --num-workers 4

# Example 5: Export all cameras (one sample per camera per episode)
python scripts/prepare_lerobot_vqa.py \
    --lerobot-dataset /path/to/lerobot/dataset \
    --enumerate \
    --as-video \
    --export-webdataset \
    --camera-key-mode all \
    --webdataset-dir /output/all_cameras \
    --samples-per-shard 100

echo "Webdataset export examples complete!"
echo ""
echo "Output format:"
echo "  Each tar shard contains samples with:"
echo "    - __key__: unique identifier (e.g., 'bridge_episode_000042_observation_images_camera_0')"
echo "    - mp4: video bytes (H.264 encoded MP4)"
echo "    - json: JSON with conversations array containing Q&A pairs"
echo ""
echo "To load the webdataset:"
echo "  import webdataset as wds"
echo "  dataset = wds.WebDataset('/output/webdataset/*.tar')"
echo "  for sample in dataset:"
echo "      key = sample['__key__']"
echo "      video_bytes = sample['mp4']"
echo "      metadata = json.loads(sample['json'])"
echo "      conversations = metadata['conversations']"
