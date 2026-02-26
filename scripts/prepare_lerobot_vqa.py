#!/usr/bin/env python3
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
Prepare a VQA dataset that uses ``lerobot://`` references for images/videos.

This script takes an existing VQA JSON/JSONL file (ShareGPT format) and a
LeRobot dataset, then creates a new dataset file where the ``images`` column
contains ``lerobot://<frame_index>::<camera_key>`` references instead of
local file paths.

At training time, LLaMA Factory resolves these references on-the-fly during
collation — **no frames are extracted to disk**.

Usage Examples
--------------

1) Map a VQA dataset where each row already has a ``lerobot_index`` field::

    python scripts/prepare_lerobot_vqa.py \\
        --vqa-input my_vqa.jsonl \\
        --output my_vqa_lerobot.jsonl \\
        --lerobot-dataset /path/to/lerobot/dataset \\
        --camera-key observation.images.front \\
        --index-field lerobot_index

2) Map a VQA dataset where each row has an ``episode_id`` + ``frame_id``::

    python scripts/prepare_lerobot_vqa.py \\
        --vqa-input my_vqa.jsonl \\
        --output my_vqa_lerobot.jsonl \\
        --lerobot-dataset /path/to/lerobot/dataset \\
        --camera-key observation.images.front \\
        --episode-field episode_id \\
        --frame-field frame_id

3) Create a VQA dataset from scratch (enumerate all frames)::

    python scripts/prepare_lerobot_vqa.py \\
        --lerobot-dataset /path/to/lerobot/dataset \\
        --output enumerate_vqa.jsonl \\
        --camera-key observation.images.front \\
        --enumerate \\
        --question-template "Describe what the robot is doing in this frame."

4) Multi-dataset mode — each row has a ``dataset`` field::

    python scripts/prepare_lerobot_vqa.py \\
        --vqa-input my_multi_vqa.jsonl \\
        --output my_multi_lerobot.jsonl \\
        --lerobot-datasets '{"bridge": "/data/bridge", "droid": "/data/droid"}' \\
        --dataset-field dataset \\
        --camera-keys '{"bridge": "observation.images.front", "droid": "cam_high"}' \\
        --index-field lerobot_index

5) Enumerate all frames across multiple datasets::

    python scripts/prepare_lerobot_vqa.py \\
        --lerobot-datasets '{"bridge": "/data/bridge", "droid": "/data/droid"}' \\
        --output multi_enumerate.jsonl \\
        --camera-keys '{"bridge": "observation.images.front", "droid": "cam_high"}' \\
        --enumerate \\
        --question-template "Describe what the robot is doing."

6) Enumerate episodes as videos — one row per episode, ``videos`` column::

    python scripts/prepare_lerobot_vqa.py \\
        --lerobot-dataset /path/to/lerobot/dataset \\
        --output episodes_vqa.jsonl \\
        --camera-key observation.images.front \\
        --enumerate --as-video \\
        --question-template "Describe what the robot is doing in this episode."

    Produces: {"videos": ["lerobot://episode:5::observation.images.front"], ...}

7) Convert existing VQA (each row has ``episode_id``) to episode video refs::

    python scripts/prepare_lerobot_vqa.py \\
        --vqa-input my_vqa.jsonl \\
        --output episodes_lerobot.jsonl \\
        --camera-key observation.images.front \\
        --episode-field episode_id \\
        --as-video

    With multi-dataset + named episodes (e.g. oxe)::

        --lerobot-datasets '{"oxe": "/data/oxe"}' --dataset-field dataset \\
        --camera-keys '{"oxe": "cam_key"}' --episode-field episode_id --as-video

    Produces: {"videos": ["lerobot://oxe::episode:5::cam_key"], ...}

After creating the dataset, add an entry to ``data/dataset_info.json``::

    "my_lerobot_vqa": {
        "file_name": "/path/to/my_vqa_lerobot.jsonl",
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "images"
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant"
        }
    }

(Include ``"videos": "videos"`` in ``columns`` if your JSONL has video refs too.)

Then train with::

    # Single dataset
    export LEROBOT_DATASET=/path/to/lerobot/dataset
    export LEROBOT_CAMERA_KEY=observation.images.front

    # Multi-dataset
    export LEROBOT_DATASETS='{"bridge": "/data/bridge", "droid": "/data/droid"}'
    export LEROBOT_CAMERA_KEYS='{"bridge": "observation.images.front", "droid": "cam_high"}'

    llamafactory-cli train --config your_config.yaml
"""

import argparse
import json
import random
import sys
from pathlib import Path

from tqdm import tqdm

def load_jsonl(path: str):
    """Load a JSONL file, yielding one dict per line."""
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_json_or_jsonl(path: str) -> list[dict]:
    """Load JSON (array) or JSONL file."""
    with open(path) as f:
        first_char = f.read(1)
    if first_char == "[":
        with open(path) as f:
            return json.load(f)
    else:
        return list(load_jsonl(path))


def make_lerobot_ref(index: int, camera_key: str, dataset_name: str = "") -> str:
    """Build a lerobot:// reference string.

    When *dataset_name* is given (e.g. ``"bridge"``), the URI is fully
    explicit so the bridge can resolve it to the correct dataset.
    """
    if dataset_name:
        return f"lerobot://{dataset_name}::{index}::{camera_key}"
    return f"lerobot://{index}::{camera_key}"


def make_lerobot_episode_ref(episode_idx: int, camera_key: str, dataset_name: str = "") -> str:
    """Build a ``lerobot://episode:<ep>::<cam>`` reference for a whole episode.

    The resulting URI goes in the ``videos`` column of a ShareGPT row.  At
    collation time, LLaMA Factory calls :func:`load_lerobot_video_frames`,
    which loads every frame of the episode from the LeRobot MP4 videos.

    With *dataset_name* (multi-dataset setup)::

        lerobot://oxe::episode:5::cam_key

    Without *dataset_name* (single-dataset, ``LEROBOT_DATASET`` env var)::

        lerobot://episode:5::observation.images.front
    """
    if dataset_name:
        return f"lerobot://{dataset_name}::episode:{episode_idx}::{camera_key}"
    return f"lerobot://episode:{episode_idx}::{camera_key}"


def resolve_camera_keys(ds, user_keys: list[str] | None) -> list[str]:
    """Return the effective list of camera keys for a dataset.

    If *user_keys* is ``None`` or empty, falls back to ``ds.meta.video_keys``.
    Raises ``ValueError`` if neither source yields any keys.
    """
    if user_keys:
        return list(user_keys)
    keys = list(ds.meta.video_keys)
    if not keys:
        raise ValueError(
            "Dataset has no video_keys and no --camera-key was provided. "
            "Pass at least one key explicitly."
        )
    print(f"  No camera keys specified — using all video_keys: {keys}")
    return keys


def convert_existing_vqa(
    vqa_data: list[dict],
    camera_key: str,
    index_field: str | None = None,
    episode_field: str | None = None,
    frame_field: str | None = None,
    lerobot_ds=None,
    dataset_name: str = "",
    dataset_field: str | None = None,
    lerobot_datasets: dict[str, object] | None = None,
    camera_keys: dict[str, str] | None = None,
    as_video: bool = False,
) -> list[dict]:
    """Convert existing VQA rows to use lerobot:// references.

    The VQA dataset must have an ``images`` column (list of strings).
    Each image path is replaced by a ``lerobot://`` reference.

    The frame index is determined by:
      - ``index_field``: a direct global index into the LeRobot dataset
      - ``episode_field`` + ``frame_field``: resolved via LeRobot metadata

    Multi-dataset mode:
      - ``dataset_field``: per-row field naming the dataset (key into *lerobot_datasets*)
      - ``lerobot_datasets``: dict mapping short names to loaded LeRobotDataset objects
      - ``camera_keys``: dict mapping short names to camera keys
    """
    output = []
    for row in vqa_data:
        new_row = dict(row)

        # ── Resolve which dataset this row belongs to ──────────────────
        row_ds_name = dataset_name
        row_camera = camera_key
        row_ds = lerobot_ds

        if dataset_field and dataset_field in row:
            row_ds_name = row[dataset_field]
            if lerobot_datasets and row_ds_name in lerobot_datasets:
                row_ds = lerobot_datasets[row_ds_name]
            if camera_keys and row_ds_name in camera_keys:
                row_camera = camera_keys[row_ds_name]

        if as_video:
            # Episode video mode: use episode index directly — no global frame lookup needed.
            if not (episode_field and episode_field in row):
                raise ValueError(
                    f"--as-video requires --episode-field. Row missing '{episode_field}'."
                )
            ep_idx = int(row[episode_field])

            # Count placeholders; accept both <image> and <video> in existing templates
            num_placeholders = 0
            for msg in new_row.get("messages", new_row.get("conversations", [])):
                content = msg.get("content", msg.get("value", ""))
                num_placeholders += content.count("<video>") + content.count("<image>")

            # Normalise <image> → <video> so the VLM collator handles it correctly
            for msg in new_row.get("messages", new_row.get("conversations", [])):
                key = "content" if "content" in msg else "value"
                msg[key] = msg[key].replace("<image>", "<video>")

            new_row["videos"] = [make_lerobot_episode_ref(ep_idx, row_camera, row_ds_name)] * max(num_placeholders, 1)
            new_row.pop("images", None)
        else:
            if index_field and index_field in row:
                # Direct global index
                idx = int(row[index_field])
            elif episode_field and frame_field and episode_field in row and frame_field in row:
                # Resolve episode + frame to global index
                ep_idx = int(row[episode_field])
                frame_idx = int(row[frame_field])
                if row_ds is not None:
                    from_idx = row_ds.meta.episodes["dataset_from_index"][ep_idx]
                    idx = from_idx + frame_idx
                else:
                    raise ValueError(
                        f"LeRobot dataset must be provided for episode+frame mapping "
                        f"(dataset={row_ds_name!r})."
                    )
            else:
                raise ValueError(
                    f"Row must have either '{index_field}' or ('{episode_field}' + '{frame_field}'). "
                    f"Available keys: {list(row.keys())}"
                )

            # Count <image> placeholders in messages
            num_placeholders = 0
            for msg in new_row.get("messages", new_row.get("conversations", [])):
                content = msg.get("content", msg.get("value", ""))
                num_placeholders += content.count("<image>")

            # Build images list
            new_row["images"] = [make_lerobot_ref(idx, row_camera, row_ds_name)] * num_placeholders

        # Preserve dataset field in output for multi-dataset training
        if dataset_field and dataset_field in row:
            new_row["dataset"] = row_ds_name

        # Remove mapping fields from output (keep it clean)
        for field in [index_field, episode_field, frame_field]:
            if field and field in new_row and field != "dataset":
                del new_row[field]

        output.append(new_row)

    return output


def get_available_camera_keys(ds, frame_idx: int, candidate_keys: list[str]) -> list[str]:
    """Return the subset of *candidate_keys* that are available at *frame_idx*.

    Reads from ``ds.hf_dataset`` (plain Arrow, no tensor decoding) to avoid
    thread-safety issues with ``ds.__getitem__``.
    Availability is stored as a per-frame boolean feature named
    ``<camera_key>_available`` (e.g. ``observation.images.camera_0_available``).
    Keys with no corresponding availability feature are assumed to be available.
    Falls back to *candidate_keys* unchanged on error or if result would be empty.
    """
    hf_ds = getattr(ds, "hf_dataset", None)
    if hf_ds is None:
        return candidate_keys
    try:
        row = hf_ds[frame_idx]
    except Exception:
        return candidate_keys

    available = []
    for key in candidate_keys:
        avail_field = f"{key}_available"
        if avail_field in row:
            val = row[avail_field]
            if hasattr(val, "item"):
                val = val.item()
            if val:
                available.append(key)
        else:
            available.append(key)  # no availability flag → assume present

    return available if available else candidate_keys


def build_episode_camera_availability(
    ds,
    episode_indices,
    candidate_keys: list[str],
    num_workers: int = 16,  # kept for API compatibility, no longer used
) -> dict[int, list[str]]:
    """Pre-compute available camera keys for each episode.

    Reads availability from ``ds.hf_dataset`` (Arrow, not PyTorch) in a single
    batched select over the first frame of each episode.  This is both
    thread-safe and fast — no per-frame ``ds.__getitem__`` calls.

    If the dataset has no ``_available`` columns, returns all candidate keys
    for every episode without touching any data.
    """
    episode_list = list(episode_indices)
    all_available = {ep_idx: list(candidate_keys) for ep_idx in episode_list}

    hf_ds = getattr(ds, "hf_dataset", None)
    if hf_ds is None:
        return all_available

    # Determine which availability fields actually exist in the Arrow schema
    avail_fields = [f"{key}_available" for key in candidate_keys]
    existing_fields = [f for f in avail_fields if f in hf_ds.column_names]
    if not existing_fields:
        return all_available  # dataset has no availability columns

    # Collect first-frame index for each episode
    ep_to_first: dict[int, int] = {}
    for ep_idx in episode_list:
        try:
            ep_to_first[ep_idx] = ds.meta.episodes[ep_idx]["dataset_from_index"]
        except Exception:
            pass  # episode not found — will keep default (all keys)

    if not ep_to_first:
        return all_available

    # Read one Arrow row per episode (memory-mapped seek, no bulk copy).
    # hf_dataset.__getitem__(int) returns a plain dict from the Arrow table —
    # safe, cheap, and does not load the full dataset into memory.
    for ep_idx in tqdm(list(ep_to_first.keys()), desc="Building camera availability", leave=False):
        first_frame_idx = ep_to_first[ep_idx]
        try:
            row = hf_ds[first_frame_idx]
        except Exception:
            continue  # keep default: all keys

        available = []
        for key in candidate_keys:
            field = f"{key}_available"
            if field in row:
                val = row[field]
                if hasattr(val, "item"):
                    val = val.item()
                if val:
                    available.append(key)
            else:
                available.append(key)
        all_available[ep_idx] = available if available else list(candidate_keys)

    return all_available


def get_episode_task(lerobot_ds, ep_idx: int) -> str:
    """Return the language instruction for *ep_idx*, or '' if unavailable.

    LeRobot stores per-episode tasks in ``meta.episodes[ep_idx]["tasks"]`` as a
    list of strings (usually one entry).  Multiple tasks are joined with ``"; "``.
    """
    try:
        ep = lerobot_ds.meta.episodes[ep_idx]
        tasks = ep.get("tasks", []) if isinstance(ep, dict) else []
        if tasks:
            return "; ".join(str(t) for t in tasks if t)
    except Exception:
        pass
    return ""


def build_frame_to_episode(lerobot_ds) -> dict[int, int]:
    """Build a mapping ``{global_frame_idx: episode_idx}`` from episode metadata."""
    mapping: dict[int, int] = {}
    for ep_idx in range(lerobot_ds.num_episodes):
        try:
            ep = lerobot_ds.meta.episodes[ep_idx]
            from_idx = ep["dataset_from_index"]
            to_idx = ep["dataset_to_index"]
            for i in range(from_idx, to_idx):
                mapping[i] = ep_idx
        except Exception:
            pass
    return mapping


def enumerate_lerobot_frames(
    lerobot_ds,
    camera_keys: list[str],
    camera_key_mode: str,
    question_template: str,
    episodes: list[int] | None = None,
    max_frames: int | None = None,
    dataset_name: str = "",
    num_workers: int = 16,
) -> list[dict]:
    """Create a VQA dataset by enumerating all frames in a LeRobot dataset.

    *camera_key_mode* controls how multiple camera keys are handled:

    - ``"all"``    — one row per (frame × camera key)
    - ``"sample"`` — one row per frame, camera key sampled uniformly at random

    When *dataset_name* is given, the ``"dataset"`` field is included in
    each row and the URI is fully explicit.
    """
    output = []
    total = lerobot_ds.num_frames

    if episodes is not None:
        frame_indices = []
        for ep_idx in episodes:
            from_idx = lerobot_ds.meta.episodes[ep_idx]["dataset_from_index"]
            to_idx = lerobot_ds.meta.episodes[ep_idx]["dataset_to_index"]
            frame_indices.extend(range(from_idx, to_idx))
    else:
        frame_indices = range(total)

    if max_frames is not None:
        frame_indices = list(frame_indices)[:max_frames]

    frame_to_ep = build_frame_to_episode(lerobot_ds)

    # Pre-compute per-episode availability in parallel (one probe per episode)
    episode_set = {frame_to_ep[i] for i in frame_indices if i in frame_to_ep}
    ep_avail = build_episode_camera_availability(lerobot_ds, episode_set, camera_keys, num_workers)

    for idx in tqdm(frame_indices, desc="Enumerating frames"):
        ep_idx = frame_to_ep.get(idx)
        language_instruction = get_episode_task(lerobot_ds, ep_idx) if ep_idx is not None else ""

        # Look up pre-computed availability — no ds[idx] call in the hot loop
        avail_keys = ep_avail.get(ep_idx, camera_keys) if ep_idx is not None else camera_keys

        if camera_key_mode == "sample":
            keys_for_row = [random.choice(avail_keys)]
        else:  # "all"
            keys_for_row = avail_keys

        for cam in keys_for_row:
            messages = [{"role": "user", "content": f"<image>{question_template}"}]
            if language_instruction:
                messages.append({"role": "assistant", "content": language_instruction})
            row = {
                "messages": messages,
                "images": [make_lerobot_ref(idx, cam, dataset_name)],
                "camera_key": cam,
            }
            if dataset_name:
                row["dataset"] = dataset_name
            output.append(row)

    return output


def enumerate_lerobot_episodes(
    lerobot_ds,
    camera_keys: list[str],
    camera_key_mode: str,
    question_template: str,
    episodes: list[int] | None = None,
    dataset_name: str = "",
    num_workers: int = 16,
) -> list[dict]:
    """Create a VQA dataset with one row per episode, each treated as a video.

    *camera_key_mode* controls how multiple camera keys are handled:

    - ``"all"``    — one row per (episode × camera key)
    - ``"sample"`` — one row per episode, camera key sampled uniformly at random

    Each row has a ``<video>`` placeholder and the episode reference in the
    ``videos`` column.  At collation time, :func:`load_lerobot_video_frames`
    loads all frames of the episode.
    """
    output = []
    episode_indices = list(episodes) if episodes is not None else range(lerobot_ds.num_episodes)

    # Pre-compute per-episode availability in parallel
    ep_avail = build_episode_camera_availability(lerobot_ds, episode_indices, camera_keys, num_workers)

    for ep_idx in tqdm(episode_indices, desc="Enumerating episodes"):
        language_instruction = get_episode_task(lerobot_ds, ep_idx)

        avail_keys = ep_avail.get(ep_idx, camera_keys)

        if camera_key_mode == "sample":
            keys_for_row = [random.choice(avail_keys)]
        else:  # "all"
            keys_for_row = avail_keys

        for cam in keys_for_row:
            messages = [{"role": "user", "content": f"<video>{question_template}"}]
            if language_instruction:
                messages.append({"role": "assistant", "content": language_instruction})
            row = {
                "messages": messages,
                "videos": [make_lerobot_episode_ref(ep_idx, cam, dataset_name)],
                "camera_key": cam,
            }
            if dataset_name:
                row["dataset"] = dataset_name
            output.append(row)

    return output


def _source_name_from_args(args) -> str:
    """Derive a short, filesystem-safe source name from dataset arguments."""
    ds_map_raw = getattr(args, "lerobot_datasets", None)
    if ds_map_raw:
        ds_map = json.loads(ds_map_raw) if isinstance(ds_map_raw, str) else ds_map_raw
        return "_".join(sorted(ds_map.keys()))
    elif getattr(args, "lerobot_dataset", None):
        # Last path component for local paths; replace slashes for repo IDs
        p = args.lerobot_dataset.rstrip("/")
        return Path(p).name.replace("-", "_")
    elif getattr(args, "vqa_input", None):
        return Path(args.vqa_input).stem
    return "dataset"


def _build_output_path(args) -> str:
    """Auto-generate a structured output path when --output is not provided.

    Scheme::

        <base_dir>/<source>/<mode>/<split>.jsonl

    - ``source``  : derived from dataset name(s) or vqa-input stem
    - ``mode``    : ``scratch`` (--enumerate) | ``from_vqa`` (--vqa-input)
    - ``split``   : ``train`` | ``val`` | ``test``  (from --split)
    """
    if args.output:
        return args.output
    base = args.base_dir
    source = _source_name_from_args(args)
    mode = "scratch" if args.enumerate else "from_vqa"
    split = args.split
    return str(Path(base) / source / mode / f"{split}.jsonl")


def main():
    parser = argparse.ArgumentParser(description="Prepare VQA dataset with lerobot:// references")
    parser.add_argument("--vqa-input", type=str, help="Input VQA JSON/JSONL file to convert")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help=(
            "Output JSONL file path. If omitted, auto-generated as "
            "<base-dir>/<source>/<mode>/<split>.jsonl"
        ),
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default="/data/robogrounder/processed_datasets",
        help="Root directory for auto-generated output paths (default: /data/robogrounder/processed_datasets)",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "val", "test"],
        default="train",
        help="Dataset split label used in the auto-generated output path (default: train)",
    )

    # Single dataset
    parser.add_argument("--lerobot-dataset", type=str, help="LeRobot dataset path or repo_id (single dataset)")
    parser.add_argument(
        "--camera-key",
        type=str,
        nargs="+",
        default=None,
        metavar="KEY",
        help=(
            "One or more LeRobot camera keys for single-dataset mode. "
            "If omitted, all keys from dataset.meta.video_keys are used."
        ),
    )
    parser.add_argument(
        "--camera-key-mode",
        type=str,
        choices=["all", "sample"],
        default="sample",
        help=(
            "How to handle multiple camera keys per frame/episode. "
            "'all': one row per (frame × key). "
            "'sample': one row per frame, one key sampled at random. "
            "(default: all)"
        ),
    )

    # Multi-dataset
    parser.add_argument(
        "--lerobot-datasets",
        type=str,
        help='JSON dict mapping short names to paths/repo_ids, e.g. \'{"bridge": "/data/bridge", "droid": "/data/droid"}\'',
    )
    parser.add_argument(
        "--dataset-field",
        type=str,
        help="Field in each VQA row that names the dataset (e.g. 'dataset')",
    )
    parser.add_argument(
        "--camera-keys",
        type=str,
        help=(
            'JSON dict mapping dataset names to camera key(s). '
            'Values can be a single string or a list of strings. '
            'E.g. \'{"bridge": "observation.images.front", "droid": ["cam_high", "cam_low"]}\''
        ),
    )

    # For existing VQA conversion
    parser.add_argument("--index-field", type=str, help="Field name for global LeRobot frame index")
    parser.add_argument("--episode-field", type=str, help="Field name for episode index")
    parser.add_argument("--frame-field", type=str, help="Field name for frame index within episode")

    # For enumeration mode
    parser.add_argument("--enumerate", action="store_true", help="Enumerate all frames in LeRobot dataset(s)")
    parser.add_argument("--question-template", type=str, default="Describe what is happening in this image.")
    parser.add_argument("--episodes", type=int, nargs="*", help="Specific episodes to enumerate")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to include")
    parser.add_argument("--num-workers", type=int, default=16, help="Number of threads for parallel camera availability probing (default: 16)")
    parser.add_argument(
        "--as-video",
        action="store_true",
        help=(
            "Output episode video references (``videos`` column) instead of per-frame image "
            "references (``images`` column). "
            "With --enumerate, produces one row per episode. "
            "With --vqa-input and --episode-field, converts episode indices to video URIs "
            "without needing to load the LeRobot dataset."
        ),
    )

    args = parser.parse_args()
    
        # python scripts/prepare_lerobot_vqa.py \\
        # --lerobot-dataset /path/to/lerobot/dataset \\
        # --output episodes_vqa.jsonl \\
        # --camera-key observation.images.front \\
        # --enumerate --as-video \\
        # --question-template "Describe what the robot is doing in this episode."
        
    args.lerobot_datasets = json.dumps({"bridge": "/data/jnogga/bridge_data_v2_teleop"})
    #args.camera_keys = json.dumps({"bridge": "observation.images.camera_0"})
    args.enumerate = True
    args.as_video = True
    
    args.question_template = "What task did the robot perform in this video?"
    
    
    args.output = _build_output_path(args)

    #

    # ── Parse multi-dataset config ────────────────────────────────────────
    datasets_map: dict[str, str] = {}
    camera_keys_map: dict[str, list[str]] = {}  # values are always lists
    if args.lerobot_datasets:
        datasets_map = json.loads(args.lerobot_datasets)
    if args.camera_keys:
        raw_cam_keys = json.loads(args.camera_keys)
        # Normalise: values may be a single string or a list of strings
        camera_keys_map = {
            k: ([v] if isinstance(v, str) else list(v)) for k, v in raw_cam_keys.items()
        }

    multi_mode = bool(datasets_map)

    # ── Load LeRobot dataset(s) for metadata ──────────────────────────────
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    import os

    lerobot_ds = None
    lerobot_datasets: dict[str, object] = {}

    def _load_ds(path_or_repo: str):
        if os.path.isdir(path_or_repo):
            return LeRobotDataset(repo_id="local", root=path_or_repo)
        else:
            return LeRobotDataset(repo_id=path_or_repo)

    if multi_mode:
        # as_video conversion needs no metadata (episode idx used directly); enumerate always does
        need_metadata = (args.episode_field and not args.as_video) or args.enumerate
        if need_metadata:
            for name, path in datasets_map.items():
                ds = _load_ds(path)
                lerobot_datasets[name] = ds
                print(f"LeRobot dataset '{name}': {ds.num_episodes} episodes, {ds.num_frames} frames")
    elif args.lerobot_dataset:
        if (args.episode_field and not args.as_video) or args.enumerate:
            lerobot_ds = _load_ds(args.lerobot_dataset)
            print(f"LeRobot dataset: {lerobot_ds.num_episodes} episodes, {lerobot_ds.num_frames} frames")

    # ── Enumeration mode ──────────────────────────────────────────────────
    if args.enumerate:
        if multi_mode:
            # Enumerate across all datasets
            output_data = []
            for name, path in datasets_map.items():
                ds = lerobot_datasets.get(name) or _load_ds(path)
                cam_keys = resolve_camera_keys(ds, camera_keys_map.get(name) or args.camera_key)
                if args.as_video:
                    rows = enumerate_lerobot_episodes(
                        ds,
                        camera_keys=cam_keys,
                        camera_key_mode=args.camera_key_mode,
                        question_template=args.question_template,
                        episodes=args.episodes,
                        dataset_name=name,
                        num_workers=args.num_workers,
                    )
                else:
                    rows = enumerate_lerobot_frames(
                        ds,
                        camera_keys=cam_keys,
                        camera_key_mode=args.camera_key_mode,
                        question_template=args.question_template,
                        episodes=args.episodes,
                        max_frames=args.max_frames,
                        dataset_name=name,
                        num_workers=args.num_workers,
                    )
                output_data.extend(rows)
                print(f"  {name}: {len(rows)} rows")
        else:
            if not lerobot_ds:
                parser.error("--lerobot-dataset or --lerobot-datasets is required for --enumerate.")
            cam_keys = resolve_camera_keys(lerobot_ds, args.camera_key)
            if args.as_video:
                output_data = enumerate_lerobot_episodes(
                    lerobot_ds,
                    camera_keys=cam_keys,
                    camera_key_mode=args.camera_key_mode,
                    question_template=args.question_template,
                    episodes=args.episodes,
                    num_workers=args.num_workers,
                )
            else:
                output_data = enumerate_lerobot_frames(
                    lerobot_ds,
                    camera_keys=cam_keys,
                    camera_key_mode=args.camera_key_mode,
                    question_template=args.question_template,
                    episodes=args.episodes,
                    max_frames=args.max_frames,
                    num_workers=args.num_workers,
                )

    # ── Conversion mode ───────────────────────────────────────────────────
    elif args.vqa_input:
        vqa_data = load_json_or_jsonl(args.vqa_input)
        # convert_existing_vqa maps each row to a single camera key; use the first
        # provided key (or let it default to None if unset — caller must ensure
        # camera key is resolvable via camera_keys_map or index_field).
        single_camera_key = args.camera_key[0] if args.camera_key else ""
        # Flatten list values in camera_keys_map → single string (first key)
        flat_camera_keys_map = {k: v[0] for k, v in camera_keys_map.items()} if camera_keys_map else None
        output_data = convert_existing_vqa(
            vqa_data,
            camera_key=single_camera_key,
            index_field=args.index_field,
            episode_field=args.episode_field,
            frame_field=args.frame_field,
            lerobot_ds=lerobot_ds,
            dataset_field=args.dataset_field,
            lerobot_datasets=lerobot_datasets if multi_mode else None,
            camera_keys=flat_camera_keys_map if multi_mode else None,
            as_video=args.as_video,
        )
    else:
        parser.error("Either --vqa-input or --enumerate is required.")
        sys.exit(1)

    # Write output
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for row in output_data:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Written {len(output_data)} rows to {args.output}")


if __name__ == "__main__":
    main()
