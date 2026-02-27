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
Bridge for loading frames from LeRobot v3 datasets.

This module provides lazy, on-the-fly frame loading from LeRobot datasets
without extracting frames to disk. Frames are decoded at collation time
(per-batch) by the DataLoader workers, making it memory-efficient and
compatible with resource-constrained clusters.

Single-dataset usage:
    1. Set environment variables:
        export LEROBOT_DATASET=/path/to/lerobot/dataset  (or hub repo_id)
        export LEROBOT_CAMERA_KEY=observation.images.front  (optional)

    2. In your VQA dataset JSON, use lerobot:// references in the images column:
        {"images": ["lerobot://42"], ...}                  # frame index 42, default camera
        {"images": ["lerobot://42::observation.images.wrist"], ...}  # explicit camera
        {"images": ["lerobot:///path/to/ds::42::cam_key"], ...}      # explicit dataset path

    3. These references flow as lightweight strings through tokenization and are
       resolved to PIL images only at collation time (zero intermediate disk I/O).

Multi-dataset usage (e.g. bridge, droid, oxe):
    1. Set LEROBOT_DATASETS to a JSON mapping of short names to paths/repo_ids:
        export LEROBOT_DATASETS='{"bridge": "/data/bridge_v2", "droid": "lerobot/droid", "oxe": "/data/oxe"}'

    2. Optionally set per-dataset camera keys:
        export LEROBOT_CAMERA_KEYS='{"bridge": "observation.images.front", "droid": "cam_high"}'
        export LEROBOT_VIDEO_KEYS='{"bridge": "observation.images.front", "droid": "cam_high"}'

    3. Use the short name in lerobot:// URIs:
        {"images": ["lerobot://bridge::42::observation.images.front"], ...}
        {"images": ["lerobot://droid::100"], ...}   # per-dataset default camera
        {"videos": ["lerobot://oxe::episode:5::cam_key"], ...}
"""

import json
import os
import threading
from typing import TYPE_CHECKING, Optional

from ..extras import logging


if TYPE_CHECKING:
    from PIL.Image import Image as ImageObject


logger = logging.get_logger(__name__)

LEROBOT_PREFIX = "lerobot://"

# ── Module-level dataset cache (per-process, thread-safe) ──────────────────
_dataset_cache: dict[str, object] = {}
_cache_lock = threading.Lock()

# ── One-shot preload trigger: fire preload_all_datasets() on first real use ─
_preload_all_triggered: bool = False
_preload_trigger_lock = threading.Lock()

# ── Multi-dataset name → path mapping (parsed once from env) ───────────────
_datasets_map: dict[str, str] | None = None
_camera_keys_map: dict[str, str] | None = None
_video_keys_map: dict[str, str] | None = None


def _get_datasets_map() -> dict[str, str]:
    """Parse ``LEROBOT_DATASETS`` env var (JSON dict: name → path/repo_id).

    Example::

        export LEROBOT_DATASETS='{"bridge": "/data/bridge_v2", "droid": "lerobot/droid"}'
    """
    global _datasets_map
    if _datasets_map is not None:
        return _datasets_map
    raw = os.environ.get("LEROBOT_DATASETS", "")
    if raw:
        _datasets_map = json.loads(raw)
        logger.info_rank0(
            f"Loaded LEROBOT_DATASETS mapping with {len(_datasets_map)} entries: "
            f"{list(_datasets_map.keys())}"
        )
    else:
        _datasets_map = {}
    return _datasets_map


def resolve_dataset_path(name_or_path: str) -> str:
    """Resolve a dataset short name to its actual path/repo_id.

    Checks the ``LEROBOT_DATASETS`` JSON mapping first; if the key is not
    found the value is returned as-is (assumed to be a direct path or Hub
    repo_id).
    """
    mapping = _get_datasets_map()
    if name_or_path in mapping:
        return mapping[name_or_path]
    return name_or_path


def resolve_camera_key(dataset_name: str = "", is_video: bool = False) -> str:
    """Resolve the camera key for a dataset, with per-dataset overrides.

    Resolution order:
        1. ``LEROBOT_VIDEO_KEYS`` JSON mapping (if *is_video*, keyed by dataset name)
        2. ``LEROBOT_CAMERA_KEYS`` JSON mapping (keyed by dataset name)
        3. ``LEROBOT_VIDEO_KEY`` / ``LEROBOT_CAMERA_KEY`` env var (global default)
        4. Hardcoded default: ``observation.images.front``
    """
    global _camera_keys_map, _video_keys_map

    if is_video:
        if _video_keys_map is None:
            raw = os.environ.get("LEROBOT_VIDEO_KEYS", "")
            _video_keys_map = json.loads(raw) if raw else {}
        if dataset_name and dataset_name in _video_keys_map:
            return _video_keys_map[dataset_name]
        # Fall through to camera keys for video (video defaults to camera key)

    if _camera_keys_map is None:
        raw = os.environ.get("LEROBOT_CAMERA_KEYS", "")
        _camera_keys_map = json.loads(raw) if raw else {}
    if dataset_name and dataset_name in _camera_keys_map:
        return _camera_keys_map[dataset_name]

    if is_video:
        env_key = os.environ.get("LEROBOT_VIDEO_KEY", "")
        if env_key:
            return env_key

    return os.environ.get("LEROBOT_CAMERA_KEY", "observation.images.front")


def get_available_datasets() -> dict[str, str]:
    """Return the name → path mapping from ``LEROBOT_DATASETS``, if configured."""
    return dict(_get_datasets_map())


def is_lerobot_reference(ref: object) -> bool:
    """Check whether *ref* is a ``lerobot://`` reference string."""
    return isinstance(ref, str) and ref.startswith(LEROBOT_PREFIX)


def parse_lerobot_reference(ref: str) -> tuple[str, int, str]:
    """Parse a ``lerobot://`` reference into (dataset_id, frame_index, camera_key).

    The returned ``dataset_id`` may be a short name (e.g. ``"bridge"``), a local
    path, or a Hub repo_id.  It is resolved to an actual path lazily by
    :func:`_get_lerobot_dataset` via :func:`resolve_dataset_path`.

    Supported formats (separator is ``::``):
        - ``lerobot://<index>``                                → env-var dataset + per-dataset camera
        - ``lerobot://<index>::<camera_key>``                  → env-var dataset
        - ``lerobot://<name_or_path>::<index>``                → named/path dataset + per-dataset camera
        - ``lerobot://<name_or_path>::<index>::<camera_key>``  → fully explicit
    """
    body = ref[len(LEROBOT_PREFIX) :]
    parts = body.split("::")

    if len(parts) == 1:
        # lerobot://42
        index_str = parts[0]
        dataset_id = os.environ.get("LEROBOT_DATASET", "")
        camera_key = resolve_camera_key(dataset_id)
    elif len(parts) == 2:
        # lerobot://42::observation.images.wrist  OR  lerobot://bridge::42
        try:
            index_str = parts[0]
            int(index_str)  # test if first part is an integer
            camera_key = parts[1]
            dataset_id = os.environ.get("LEROBOT_DATASET", "")
        except ValueError:
            dataset_id = parts[0]
            index_str = parts[1]
            camera_key = resolve_camera_key(dataset_id)
    elif len(parts) == 3:
        # lerobot://bridge::42::observation.images.front
        dataset_id, index_str, camera_key = parts
    else:
        raise ValueError(f"Invalid lerobot reference format: {ref!r}")

    if not dataset_id:
        raise ValueError(
            "LEROBOT_DATASET environment variable must be set when using short-form lerobot:// references. "
            "Alternatively, set LEROBOT_DATASETS with a JSON mapping of dataset names to paths. "
            "Example:  export LEROBOT_DATASET=/path/to/lerobot/dataset"
        )

    return dataset_id, int(index_str), camera_key


def _get_lerobot_dataset(dataset_id: str):
    """Return a cached ``LeRobotDataset`` instance (lazy-init, thread-safe).

    ``dataset_id`` may be a short name from the ``LEROBOT_DATASETS`` mapping
    (e.g. ``"bridge"``), a local filesystem path, or a Hub ``repo_id``.
    Short names are resolved via :func:`resolve_dataset_path` before loading.
    """
    global _preload_all_triggered
    resolved = resolve_dataset_path(dataset_id)

    if resolved in _dataset_cache:
        return _dataset_cache[resolved]

    with _cache_lock:
        # Double-check after acquiring lock
        if resolved in _dataset_cache:
            return _dataset_cache[resolved]

        try:
            from lerobot.datasets.lerobot_dataset import LeRobotDataset
        except ImportError:
            raise ImportError(
                "The `lerobot` package is required to use lerobot:// references. "
                "Install it with: pip install lerobot"
            )

        log_name = f"{dataset_id}" if dataset_id == resolved else f"{dataset_id} → {resolved}"
        logger.info_rank0(f"Initializing LeRobot dataset: {log_name}")
        if os.path.isdir(resolved):
            ds = LeRobotDataset(repo_id="local", root=resolved)
        else:
            ds = LeRobotDataset(repo_id=resolved)

        _dataset_cache[resolved] = ds
        logger.info_rank0(
            f"LeRobot dataset loaded ({dataset_id}): {ds.num_episodes} episodes, "
            f"{ds.num_frames} frames, cameras: {ds.meta.camera_keys}"
        )
        result = ds

    # First real lerobot:// reference in this process: eagerly preload every
    # other dataset listed in the mapping so all workers hit cache hits from
    # here on.  Done *outside* _cache_lock to avoid deadlocks when
    # preload_all_datasets → _get_lerobot_dataset re-enters this function.
    if not _preload_all_triggered:
        with _preload_trigger_lock:
            if not _preload_all_triggered:
                _preload_all_triggered = True
                preload_all_datasets()

    return result


def preload_all_datasets() -> None:
    """Eagerly load and cache all LeRobot datasets declared in env vars.

    Reads every entry from ``LEROBOT_DATASETS`` (JSON name→path mapping) and,
    if set, the single-dataset ``LEROBOT_DATASET`` variable, then calls
    :func:`_get_lerobot_dataset` for each so the dataset instance is cached
    before any batch arrives.

    Call this function once in the **main process** before training starts (the
    populated cache is inherited by fork-based DataLoader workers via COW
    semantics on Linux) **and** pass :func:`lerobot_worker_init_fn` as the
    ``worker_init_fn`` argument to ``DataLoader`` for robustness under spawn-
    based multiprocessing.
    """
    errors: list[str] = []

    # --- entries from LEROBOT_DATASETS JSON mapping ---
    mapping = _get_datasets_map()
    for name_or_path in mapping:
        try:
            _get_lerobot_dataset(name_or_path)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{name_or_path!r}: {exc}")

    # --- single-dataset LEROBOT_DATASET env var ---
    single = os.environ.get("LEROBOT_DATASET", "").strip()
    if single:
        resolved_single = resolve_dataset_path(single)
        if resolved_single not in _dataset_cache:
            try:
                _get_lerobot_dataset(single)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{single!r}: {exc}")

    if errors:
        logger.warning_rank0(
            f"preload_all_datasets: failed to load {len(errors)} dataset(s):\n"
            + "\n".join(f"  - {e}" for e in errors)
        )


def lerobot_worker_init_fn(worker_id: int) -> None:  # noqa: ARG001
    """DataLoader ``worker_init_fn`` that preloads all configured LeRobot datasets.

    Pass this to ``torch.utils.data.DataLoader(worker_init_fn=lerobot_worker_init_fn)``
    whenever ``num_workers > 0`` and the dataset contains ``lerobot://`` references.
    Each spawned worker will eagerly initialise and cache every dataset listed in
    ``LEROBOT_DATASETS`` / ``LEROBOT_DATASET`` before the first batch is requested.
    """
    preload_all_datasets()


def load_lerobot_frame(ref: str) -> "ImageObject":
    """Decode a single frame from a LeRobot dataset given a ``lerobot://`` reference.

    The frame is loaded lazily from the LeRobot dataset's MP4 videos or
    parquet-embedded images—**no intermediate files are written to disk**.

    Returns:
        A PIL ``Image`` in RGB mode.
    """
    import torch
    from PIL import Image

    dataset_id, index, camera_key = parse_lerobot_reference(ref)
    ds = _get_lerobot_dataset(dataset_id)

    item = ds[index]
    if camera_key not in item:
        available = [k for k in item if k.startswith("observation.images")]
        raise KeyError(
            f"Camera key {camera_key!r} not found in LeRobot frame {index}. "
            f"Available image keys: {available}"
        )

    frame = item[camera_key]

    if isinstance(frame, torch.Tensor):
        # LeRobot returns [C, H, W] float32 in [0, 1]
        if frame.ndim == 3:
            if frame.dtype == torch.float32:
                frame = (frame.clamp(0, 1) * 255).to(torch.uint8)
            # C, H, W → H, W, C for PIL
            frame_np = frame.permute(1, 2, 0).cpu().numpy()
            return Image.fromarray(frame_np, "RGB")
        else:
            raise ValueError(f"Expected 3D tensor [C,H,W], got shape {frame.shape}")
    elif isinstance(frame, Image.Image):
        return frame
    else:
        raise TypeError(f"Unexpected frame type from LeRobot: {type(frame)}")


def load_lerobot_video_frames(
    ref: str,
    num_frames: Optional[int] = None,
) -> list["ImageObject"]:
    """Load all frames of a LeRobot episode as a list of PIL Images.

    Reference format: ``lerobot://episode:<episode_idx>::<camera_key>``

    This is useful when you want to treat an entire episode as a "video" input.

    Args:"{'initializer': {'name': 'BBoxInit', 'dim': '${globals.SLOT_DIM}', 'n_slots': '${globals.NUM_SLOTS}', 'prepend_background': 3}, 'encoder': {'backbone': {'name': 'TimmExtractor', 'model': '${globals.DINO_MODEL}', 'features': 'vit_output', 'frozen': True, 'pretrained': True}, 'output_transform': {'name': 'networks.two_layer_mlp', 'inp_dim': '${globals.FEAT_DIM}', 'outp_dim': '${globals.SLOT_DIM}', 'hidden_dim': '${mul: ${globals.FEAT_DIM}, 2}', 'layer_norm': True}}, 'grouper': {'name': 'SlotAttention', 'inp_dim': '${globals.SLOT_DIM}', 'slot_dim': '${globals.SLOT_DIM}', 'n_iters': 2, 'use_mlp': True}, 'decoder': {'name': 'MLPDecoder', 'inp_dim': '${globals.SLOT_DIM}', 'outp_dim': '${globals.FEAT_DIM}', 'hidden_dims': [1024, 1024, 1024], 'n_patches': '${globals.NUM_PATCHES}'}, 'dynamics_predictor': None, 'predictor': {'name': 'networks.TransformerEncoder', 'dim': '${globals.SLOT_DIM}', 'n_blocks': 1, 'n_heads': 4}, 'target_encoder': None, 'latent_processor': {'first_step_corrector_args': {'n_iters': 3}}, 'mask_resizers': None, 'losses': {'loss_featrec': {'name': 'MSELoss', 'pred_dims': [0, '${globals.FEAT_DIM}']}, 'loss_ss': {'name': 'Slot_Slot_Contrastive_Loss', 'pred_key': 'processor.state', 'temperature': 0.1, 'batch_contrast': True, 'patch_inputs': False, 'keep_input_dim': True, 'robot_masking': True}}, 'loss_weights': {'loss_featrec': 1.0, 'loss_ss': 0.5}, 'input_type': 'video', 'target_type': 'features', 'target_encoder_input': None, 'visualize': True, 'eval_mode_config': None, 'visualize_every_n_steps': 100, 'masks_to_visualize': None, 'load_weights': None, 'modules_to_load': None}"
        ref: A ``lerobot://episode:<ep_idx>::<camera_key>`` reference.
        num_frames: If set, uniformly sample this many frames from the episode.

    Returns:
        A list of PIL Images (the video frames).
    """
    import numpy as np
    import torch
    from PIL import Image

    body = ref[len(LEROBOT_PREFIX) :]
    parts = body.split("::")

    # Supported formats:
    #   lerobot://episode:<ep>                              → env-var dataset + per-dataset camera
    #   lerobot://episode:<ep>::<camera_key>                → env-var dataset
    #   lerobot://<name_or_path>::episode:<ep>              → named/path dataset + per-dataset camera
    #   lerobot://<name_or_path>::episode:<ep>::<cam_key>   → fully explicit
    dataset_id = os.environ.get("LEROBOT_DATASET", "")

    if len(parts) == 1:
        # lerobot://episode:<ep>
        episode_part = parts[0]
        camera_key = resolve_camera_key(dataset_id, is_video=True)
    elif len(parts) == 2:
        if parts[0].startswith("episode:"):
            # lerobot://episode:<ep>::<camera_key>
            episode_part = parts[0]
            camera_key = parts[1]
        else:
            # lerobot://bridge::episode:<ep>  (no camera specified)
            dataset_id = parts[0]
            episode_part = parts[1]
            camera_key = resolve_camera_key(dataset_id, is_video=True)
    elif len(parts) == 3:
        if parts[0].startswith("episode:"):
            raise ValueError(f"Invalid lerobot video reference format: {ref!r}")
        # lerobot://bridge::episode:<ep>::<camera_key>
        dataset_id = parts[0]
        episode_part = parts[1]
        camera_key = parts[2]
    else:
        raise ValueError(f"Invalid lerobot video reference format: {ref!r}")

    if not episode_part.startswith("episode:"):
        raise ValueError(
            f"Video reference must contain 'episode:<idx>'. Got: {ref!r}"
        )

    episode_idx = int(episode_part.split(":", 1)[1])

    if not dataset_id:
        raise ValueError(
            "LEROBOT_DATASET environment variable must be set when using short-form lerobot:// references. "
            "Alternatively, set LEROBOT_DATASETS with a JSON mapping."
        )

    ds = _get_lerobot_dataset(dataset_id)

    # Get frame range for this episode
    from_idx = ds.meta.episodes["dataset_from_index"][episode_idx]
    to_idx = ds.meta.episodes["dataset_to_index"][episode_idx]
    total = to_idx - from_idx

    if num_frames is not None and num_frames < total:
        indices = np.linspace(0, total - 1, num_frames, dtype=int) + from_idx
    else:
        indices = range(from_idx, to_idx)

    frames = []
    for idx in indices:
        item = ds[int(idx)]
        frame = item[camera_key]
        if isinstance(frame, torch.Tensor):
            if frame.dtype == torch.float32:
                frame = (frame.clamp(0, 1) * 255).to(torch.uint8)
            frame_np = frame.permute(1, 2, 0).cpu().numpy()
            frames.append(Image.fromarray(frame_np, "RGB"))
        elif isinstance(frame, Image.Image):
            frames.append(frame)
        else:
            raise TypeError(f"Unexpected frame type: {type(frame)}")

    return frames


def get_lerobot_video_info(
    ref: str,
    num_frames: Optional[int] = None,
) -> tuple[int, float, int, int]:
    """Return video metadata for a LeRobot episode **without** decoding any frames.

    Uses only episode index ranges and ``ds.meta.shapes`` / ``ds.meta.features``
    which are pure metadata reads from ``info.json``.

    Args:
        ref: A ``lerobot://episode:<ep_idx>::<camera_key>`` reference.
        num_frames: If set, the number of frames that *would* be uniformly
            sampled (mirrors :func:`load_lerobot_video_frames` logic).

    Returns:
        A tuple ``(n_sampled_frames, duration, height, width)``.
    """
    import numpy as np

    body = ref[len(LEROBOT_PREFIX):]
    parts = body.split("::")

    dataset_id = os.environ.get("LEROBOT_DATASET", "")

    if len(parts) == 1:
        episode_part = parts[0]
        camera_key = resolve_camera_key(dataset_id, is_video=True)
    elif len(parts) == 2:
        if parts[0].startswith("episode:"):
            episode_part = parts[0]
            camera_key = parts[1]
        else:
            dataset_id = parts[0]
            episode_part = parts[1]
            camera_key = resolve_camera_key(dataset_id, is_video=True)
    elif len(parts) == 3:
        if parts[0].startswith("episode:"):
            raise ValueError(f"Invalid lerobot video reference format: {ref!r}")
        dataset_id = parts[0]
        episode_part = parts[1]
        camera_key = parts[2]
    else:
        raise ValueError(f"Invalid lerobot video reference format: {ref!r}")

    if not episode_part.startswith("episode:"):
        raise ValueError(f"Video reference must contain 'episode:<idx>'. Got: {ref!r}")

    episode_idx = int(episode_part.split(":", 1)[1])

    if not dataset_id:
        raise ValueError(
            "LEROBOT_DATASET environment variable must be set when using short-form lerobot:// references."
        )

    ds = _get_lerobot_dataset(dataset_id)

    # Frame count from episode metadata
    from_idx = ds.meta.episodes["dataset_from_index"][episode_idx]
    to_idx = ds.meta.episodes["dataset_to_index"][episode_idx]
    total = to_idx - from_idx

    if num_frames is not None and num_frames < total:
        n_sampled = num_frames
    else:
        n_sampled = total

    # Resolution from dataset metadata — no frame decode needed
    shape = ds.meta.shapes[camera_key]  # (H, W, C)
    height, width = shape[0], shape[1]

    # Duration mirrors _regularize_videos: n_sampled / video_fps (caller provides fps)
    # We return 0.0 and let caller compute duration from fps + n_sampled
    duration = 0.0

    return n_sampled, duration, height, width


def clear_cache() -> None:
    """Clear all cached LeRobot dataset instances and parsed env-var mappings."""
    global _datasets_map, _camera_keys_map, _video_keys_map, _preload_all_triggered
    with _cache_lock:
        _dataset_cache.clear()
    _datasets_map = None
    _camera_keys_map = None
    _video_keys_map = None
    _preload_all_triggered = False
