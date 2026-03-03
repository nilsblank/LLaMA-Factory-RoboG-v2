# Copyright 2025 the LlamaFactory team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
LeRobot dataset loader.

Reads JSONL files containing ``lerobot_images`` / ``lerobot_videos`` references
and produces samples in the **same bytes-based format as WebDataset** (JPEG bytes
for images, MP4 bytes for videos).  This makes co-training with WebDataset
trivial — identical schemas, no special-casing in ``mm_plugin.py``.

Architecture
~~~~~~~~~~~~
1. JSONL is loaded as a **map-style** HF ``Dataset`` (cheap — just text).
2. A lazy ``set_transform`` resolves LeRobot references into bytes on access.
3. The map-style dataset is shuffled globally, then converted to an
   ``IterableDataset`` via ``to_iterable_dataset(num_shards=N)`` right before
   mixing with other datasets.  This gives true global shuffle plus native
   HF/Accelerate multi-GPU and multi-worker sharding.

JSONL format
~~~~~~~~~~~~
Each line is a JSON object with standard ``messages`` plus LeRobot references::

    {"messages": [...], "lerobot_images": [{"episode": 5, "frame": 42, "camera": "observation.images.front"}]}
    {"messages": [...], "lerobot_videos": [{"episode": 5, "camera": "observation.images.front"}]}

Image handling: LeRobot MP4 → decode → JPEG encode → bytes.  The JPEG
encode/decode adds ~1.5 ms/frame — negligible vs model forward pass — but
keeps bytes ~10× smaller in shuffle buffers / DataLoader queues.

Video handling: ffmpeg stream-copy (no decode) remuxes just the episode's time
range into a standalone in-memory MP4.  Falls back to decode-then-JPEG-list on
error.
"""

from __future__ import annotations

import io
import json
import os
from typing import TYPE_CHECKING

from ..extras import logging
from .lerobot_bridge import _clear_video_decoder_cache, _get_lerobot_dataset, resolve_dataset_path


if TYPE_CHECKING:
    pass

logger = logging.get_logger(__name__)

# Default JPEG quality for frame encoding — 95 is visually lossless, ~25 KB/frame.
_JPEG_QUALITY: int = 95


# ---------------------------------------------------------------------------
# Frame / video byte helpers
# ---------------------------------------------------------------------------


def _frame_to_jpeg_bytes(ds: object, episode: int, frame: int, camera: str) -> bytes:
    """Load a single frame from a LeRobot dataset and return JPEG bytes.

    Args:
        ds: A ``LeRobotDataset`` instance.
        episode: Episode index.
        frame: Frame offset within the episode.
        camera: Camera key (e.g. ``observation.images.front``).

    Returns:
        JPEG-encoded bytes of the frame.
    """
    import torch
    from PIL import Image

    # Compute absolute index into the dataset
    abs_idx = ds.meta.episodes[episode]["dataset_from_index"] + frame
    item = ds[abs_idx]
    _clear_video_decoder_cache()

    tensor = item[camera]
    if isinstance(tensor, torch.Tensor):
        if tensor.dtype == torch.float32:
            tensor = (tensor.clamp(0, 1) * 255).to(torch.uint8)
        # C, H, W → H, W, C
        np_frame = tensor.permute(1, 2, 0).contiguous().cpu().numpy()
        pil_img = Image.fromarray(np_frame, "RGB")
    elif isinstance(tensor, Image.Image):
        pil_img = tensor
    else:
        raise TypeError(f"Unexpected frame type from LeRobot: {type(tensor)}")

    buf = io.BytesIO()
    pil_img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
    return buf.getvalue()


def _episode_to_mp4_bytes(ds: object, episode: int, camera: str) -> bytes:
    """Remux an episode's video segment into standalone MP4 bytes via stream copy.

    Uses ffmpeg subprocess with ``-c copy`` for proper H.264 stream copy
    (preserves SPS/PPS headers, handles GOP boundaries correctly).

    Falls back to decode → JPEG list if stream copy fails.

    Args:
        ds: A ``LeRobotDataset`` instance.
        episode: Episode index.
        camera: Camera key.

    Returns:
        Standalone MP4 bytes for the episode.
    """
    try:
        return _stream_copy_episode(ds, episode, camera)
    except Exception as exc:
        logger.warning_rank0(
            f"Stream copy failed for episode {episode} camera {camera}: {exc}. "
            "Falling back to decode → JPEG list."
        )
        return _fallback_episode_jpeg_list(ds, episode, camera)


def _stream_copy_episode(ds: object, episode: int, camera: str) -> bytes:
    """Stream-copy episode packets from chunked MP4 into a standalone MP4.

    Uses ffmpeg subprocess with ``-c copy`` for proper H.264 stream copy
    (preserves SPS/PPS headers, handles GOP boundaries correctly).
    """
    import subprocess

    # LeRobot v3 meta.episodes is an HF Dataset with namespaced columns:
    #   videos/<camera>/from_timestamp, videos/<camera>/to_timestamp,
    #   videos/<camera>/chunk_index, videos/<camera>/file_index
    ep_row = ds.meta.episodes[episode]
    vid_prefix = f"videos/{camera}"
    from_ts = float(ep_row[f"{vid_prefix}/from_timestamp"])
    to_ts = float(ep_row[f"{vid_prefix}/to_timestamp"])
    chunk_idx = int(ep_row[f"{vid_prefix}/chunk_index"])
    file_idx = int(ep_row[f"{vid_prefix}/file_index"])

    # Construct path using the dataset's video_path template from info.json.
    video_path_template = ds.meta.info.get(
        "video_path",
        "videos/{video_key}/chunk-{chunk_index:03d}/file_{file_index:03d}.mp4",
    )
    rel_path = video_path_template.format(
        video_key=camera,
        chunk_index=chunk_idx,
        file_index=file_idx,
    )
    video_path = os.path.join(str(ds.root), rel_path)

    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    # Use ffmpeg -ss/-to with -c copy to remux the episode segment.
    # -ss before -i enables fast seek; -to is relative to -ss.
    duration = to_ts - from_ts
    cmd = [
        "ffmpeg",
        "-nostdin",
        "-ss", f"{from_ts:.6f}",
        "-i", video_path,
        "-t", f"{duration:.6f}",
        "-c", "copy",
        "-movflags", "+frag_keyframe+empty_moov",  # fragmented MP4 for pipe output
        "-f", "mp4",
        "pipe:1",
    ]
    result = subprocess.run(
        cmd,
        capture_output=True,
        timeout=30,
    )
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg stream copy failed: {result.stderr.decode(errors='replace')[:500]}")

    mp4_bytes = result.stdout
    if len(mp4_bytes) < 8:
        raise RuntimeError("ffmpeg produced empty or too-small MP4 output")

    return mp4_bytes


def _fallback_episode_jpeg_list(ds: object, episode: int, camera: str) -> list[bytes]:
    """Decode episode frames via ``_query_videos`` and return as a list of JPEG bytes.

    This fallback is compatible with ``mm_plugin``'s
    ``_check_video_is_nested_images`` path which treats ``list[bytes]`` as a
    sequence of individual images.
    """
    import numpy as np
    import torch
    from PIL import Image

    ep_row = ds.meta.episodes[episode]
    from_idx = ep_row["dataset_from_index"]
    to_idx = ep_row["dataset_to_index"]
    indices = list(range(from_idx, to_idx))

    if ds._absolute_to_relative_idx is not None:
        rel_indices = [ds._absolute_to_relative_idx[i] for i in indices]
        ts_raw = ds.hf_dataset[rel_indices]["timestamp"]
    else:
        ts_raw = ds.hf_dataset[indices]["timestamp"]

    if isinstance(ts_raw, torch.Tensor):
        timestamps = ts_raw.tolist()
    elif hasattr(ts_raw, "__iter__") and len(ts_raw) > 0 and isinstance(ts_raw[0], torch.Tensor):
        timestamps = [t.item() for t in ts_raw]
    else:
        timestamps = [float(t) for t in ts_raw]

    frames_dict = ds._query_videos({camera: timestamps}, episode)
    _clear_video_decoder_cache()
    frames_tensor = frames_dict[camera]
    if frames_tensor.ndim == 3:
        frames_tensor = frames_tensor.unsqueeze(0)

    frames_uint8 = (frames_tensor.clamp(0, 1) * 255).to(torch.uint8)
    np_frames = frames_uint8.permute(0, 2, 3, 1).contiguous().numpy()

    jpeg_list: list[bytes] = []
    for i in range(len(np_frames)):
        pil_img = Image.fromarray(np_frames[i], "RGB")
        buf = io.BytesIO()
        pil_img.save(buf, format="JPEG", quality=_JPEG_QUALITY)
        jpeg_list.append(buf.getvalue())

    return jpeg_list


# ---------------------------------------------------------------------------
# Sample normalization (used by set_transform)
# ---------------------------------------------------------------------------


def _normalize_lerobot_sample(
    row: dict,
    default_dataset: str,
    default_camera: str,
) -> dict:
    """Convert a JSONL row with ``lerobot_images`` / ``lerobot_videos`` into bytes-based columns.

    Output schema matches WebDataset: ``jpg`` is a list of JPEG bytes,
    ``mp4`` is a list of MP4 bytes (or nested JPEG lists on fallback).
    ``messages`` and other text columns pass through unchanged.

    Args:
        row: Parsed JSON dict from the JSONL file.
        default_dataset: Default dataset name/path if not specified per-row.
        default_camera: Default camera key if not specified per-row.

    Returns:
        Normalized dict with ``jpg`` / ``mp4`` lists and ``messages``.
    """
    result: dict = {}

    # Pass through all non-lerobot keys
    for key, value in row.items():
        if not key.startswith("lerobot_"):
            result[key] = value

    # Per-row dataset override
    row_dataset = row.get("lerobot_dataset", default_dataset)

    # --- Images ---
    lerobot_images = row.get("lerobot_images")
    if lerobot_images:
        jpg_list: list[bytes] = []
        for img_ref in lerobot_images:
            ds_name = img_ref.get("dataset", row_dataset)
            ds_path = resolve_dataset_path(ds_name)
            ds = _get_lerobot_dataset(ds_path)
            camera = img_ref.get("camera", default_camera)
            jpeg_bytes = _frame_to_jpeg_bytes(ds, img_ref["episode"], img_ref["frame"], camera)
            jpg_list.append(jpeg_bytes)
        result["jpg"] = jpg_list

    # --- Videos ---
    lerobot_videos = row.get("lerobot_videos")
    if lerobot_videos:
        mp4_list: list = []
        for vid_ref in lerobot_videos:
            ds_name = vid_ref.get("dataset", row_dataset)
            ds_path = resolve_dataset_path(ds_name)
            ds = _get_lerobot_dataset(ds_path)
            camera = vid_ref.get("camera", default_camera)
            mp4_bytes = _episode_to_mp4_bytes(ds, vid_ref["episode"], camera)
            mp4_list.append(mp4_bytes)
        result["mp4"] = mp4_list

    return result


# ---------------------------------------------------------------------------
# Map-style Dataset loading
# ---------------------------------------------------------------------------


def load_lerobot_jsonl_as_dataset(
    jsonl_files: list[str],
) -> "Dataset":
    """Load JSONL file(s) as a map-style HF ``Dataset``.

    This is cheap — it only parses JSON text (no media decoding).  The actual
    LeRobot frame/video resolution happens lazily via ``set_transform``.

    The raw JSONL columns (``messages``, ``lerobot_images``, ``lerobot_videos``,
    ``lerobot_dataset``, etc.) are stored as-is.  A subsequent ``set_transform``
    call with :func:`make_lerobot_transform` converts them to the bytes-based
    format on access.

    Args:
        jsonl_files: List of JSONL file paths.

    Returns:
        A map-style HF ``Dataset`` with the raw JSONL columns.
    """
    from datasets import Dataset

    rows: list[dict] = []
    for path in jsonl_files:
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning_rank0(f"Skipping invalid JSON at {path}:{line_num + 1}: {exc}")
                    continue
                # Serialize nested dicts/lists to JSON strings so they fit in
                # an Arrow string column (Arrow cannot store arbitrary dicts).
                for key in ("lerobot_images", "lerobot_videos", "messages"):
                    if key in row and not isinstance(row[key], str):
                        row[key] = json.dumps(row[key], ensure_ascii=False)
                rows.append(row)

    # Dataset.from_list uses the keys of the FIRST row only, so we must
    # ensure every row has a consistent set of keys (union of all keys).
    if rows:
        all_keys = set()
        for row in rows:
            all_keys.update(row.keys())
        for row in rows:
            for key in all_keys:
                row.setdefault(key, None)

    return Dataset.from_list(rows)


def make_lerobot_transform(
    default_dataset: str,
    default_camera: str,
):
    """Return a per-sample transform for ``IterableDataset.map()``.

    The returned function is called once per sample.  It deserializes JSON
    string columns back to Python objects and calls
    :func:`_normalize_lerobot_sample` to resolve LeRobot references into bytes.

    Args:
        default_dataset: Default dataset name/path.
        default_camera: Default camera key.

    Returns:
        A callable ``(example: dict) -> dict`` suitable for
        ``iterable_dataset.map(fn)``.
    """

    def _transform(example: dict) -> dict:
        # Deserialize JSON string columns back to Python objects
        row: dict = {}
        for key, val in example.items():
            if key in ("lerobot_images", "lerobot_videos", "messages") and isinstance(val, str):
                try:
                    val = json.loads(val)
                except (json.JSONDecodeError, TypeError):
                    pass
            row[key] = val

        try:
            return _normalize_lerobot_sample(row, default_dataset, default_camera)
        except Exception as exc:
            logger.warning_rank0(f"Skipping sample during transform: {exc}")
            # Pass through non-lerobot columns so the batch doesn't break.
            result = {}
            for key, val in row.items():
                if not key.startswith("lerobot_"):
                    result[key] = val
            return result

    return _transform


def _count_lerobot_samples(jsonl_files: list[str]) -> int:
    """Count total JSONL lines across files (fast, no JSON parsing).

    Used for ``max_steps`` estimation, analogous to ``_count_wds_samples``.
    """
    total = 0
    for path in jsonl_files:
        try:
            with open(path, "r", encoding="utf-8") as f:
                total += sum(1 for line in f if line.strip())
        except Exception as exc:
            logger.warning_rank0(f"Could not count samples in {path}: {exc}")
    return total
