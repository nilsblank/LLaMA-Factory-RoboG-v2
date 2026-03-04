"""
Benchmark: Lance vs WebDataset dataloader throughput
=====================================================
Uses the same helpers as the LlamaFactory training stack:
  - _load_lance_dataset        -> map-style HF Dataset (lance:// URI refs, no media in Arrow)
  - _iter_webdataset_single_tar -> schema-free TAR streaming via webdataset
  - resolve_lance_uri          -> per-process cached blob reads (O(1) after first open)

Lance note: lance is NOT fork-safe, so DataLoader workers=0 always for that path.
            WDS runs with configurable workers via IterableDataset.from_generator.

Usage:
    conda run -n roboG python benchmark_dataloader.py

    # Customise
    conda run -n roboG python benchmark_dataloader.py \
        --num-samples 5000 --workers 0 2 4 8 --no-decode
"""

from __future__ import annotations

import argparse
import io
import os
import sys
import time
import threading
from dataclasses import dataclass

import psutil
import torch
import torch.utils.data

# ---------------------------------------------------------------------------
# Make llamafactory importable without a full install
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
_SRC = os.path.join(_REPO_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
#LANCE_PATH = "/e/scratch/m3/pretrain_data/lance/eo1m.lance"
LANCE_PATH = "/e/scratch/m3/pretrain_data/lance/sav.lance"
WDS_DIR = "/e/scratch/m3/pretrain_data/wds/sav_wds"
WDS_GLOB = os.path.join(WDS_DIR, "shard-*.tar")

# LeRobot benchmark: path to the training JSONL and a name→path mapping for
# the datasets referenced inside it.  Overridable via --lerobot-* CLI flags or
# LEROBOT_DATASETS env var (same var as the training pipeline).
LEROBOT_JSONL = "/e/home/jusers/blank4/jupiter/blank4/robogrounder/bridge_droid/scratch/train_diverse_lang.jsonl"
# Default dataset roots (mirrors training env).  Can be overridden via --lerobot-datasets
# or by setting LEROBOT_DATASETS='{"droid":"/path","bridge":"/path"}' in the environment.
LEROBOT_DATASETS_DEFAULT: dict[str, str] = {
    "droid": "/e/home/jusers/blank4/jupiter/datasets/lerobot_3_0/DROID/droid_success",
    "bridge": "/e/home/jusers/blank4/jupiter/datasets/lerobot_3_0/BridgeData-V2/BridgeDataV2_teleop",
}

# ---------------------------------------------------------------------------
# Optional: PIL for image decode
# ---------------------------------------------------------------------------
try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# ---------------------------------------------------------------------------
# Optional: torchcodec (preferred) and PyAV (fallback) for video decode
# ---------------------------------------------------------------------------
try:
    from torchcodec.decoders import VideoDecoder as _VideoDecoder
    HAS_TORCHCODEC = True
except ImportError:
    HAS_TORCHCODEC = False

try:
    import av as _av
    HAS_AV = True
except ImportError:
    HAS_AV = False


# ---------------------------------------------------------------------------
# Peak RSS tracker (includes child / worker processes)
# ---------------------------------------------------------------------------
class PeakRSSTracker:
    """Context manager that samples RSS of the process tree in a background
    thread and records the peak value (in MB)."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.peak_mb: float = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._proc = psutil.Process(os.getpid())

    @staticmethod
    def _pss_mb(proc: psutil.Process) -> float:
        """Return PSS (Proportional Set Size) in MB for one process.

        PSS divides each shared page by the number of processes sharing it,
        giving a physically accurate memory attribution.  Falls back to RSS
        if /proc/smaps is unavailable (non-Linux or permission denied).
        """
        try:
            return proc.memory_full_info().pss / (1024 * 1024)
        except (AttributeError, psutil.AccessDenied, psutil.NoSuchProcess):
            try:
                return proc.memory_info().rss / (1024 * 1024)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return 0.0

    def _sample(self) -> float:
        """Return total PSS of this process + all children (MB).

        PSS avoids double-counting shared pages that forked DataLoader workers
        inherit from the parent (Python interpreter, torch libs, etc.).
        On a typical run, RSS can be 8-16x higher than PSS simply because
        each forked worker's RSS includes all shared read-only pages.
        """
        total = self._pss_mb(self._proc)
        try:
            for child in self._proc.children(recursive=True):
                total += self._pss_mb(child)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
        return total

    def _run(self):
        while not self._stop.is_set():
            mb = self._sample()
            if mb > self.peak_mb:
                self.peak_mb = mb
            self._stop.wait(self.interval)

    def __enter__(self):
        self.peak_mb = self._sample()  # baseline
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2)
        return False


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class BenchResult:
    label: str
    num_samples: int
    elapsed_s: float      # total wall time
    init_s: float         # dataset load + index prep + DataLoader setup
    iter_s: float         # pure data iteration loop
    total_bytes: int
    peak_rss_mb: float = 0.0   # peak PSS (proportional set size) — avoids double-counting shared pages
    errors: int = 0

    @property
    def samples_per_sec(self) -> float:
        return self.num_samples / self.iter_s if self.iter_s > 0 else 0.0

    @property
    def mb_per_sec(self) -> float:
        return (self.total_bytes / 1e6) / self.iter_s if self.iter_s > 0 else 0.0

    def batches_per_sec(self, batch_size: int) -> float:
        return self.samples_per_sec / batch_size if batch_size > 0 else 0.0

    def secs_per_batch(self, batch_size: int) -> float:
        bps = self.batches_per_sec(batch_size)
        return 1.0 / bps if bps > 0 else float("inf")

    def __str__(self) -> str:
        err = f"  ERRORS={self.errors}" if self.errors else ""
        return (
            f"{self.label:<58s}  "
            f"{self.num_samples:>7d} samp  "
            f"init={self.init_s:>5.2f}s  "
            f"iter={self.iter_s:>6.2f}s  "
            f"{self.samples_per_sec:>8.1f} samp/s  "
            f"{self.mb_per_sec:>7.1f} MB/s  "
            f"peak_pss={self.peak_rss_mb:>7.0f} MB"
            f"{err}"
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _decode_bytes(data: bytes) -> None:
    """Decode raw JPEG/PNG bytes via PIL (matches lazy decode in mm_plugin)."""
    if HAS_PIL:
        Image.open(io.BytesIO(data)).convert("RGB")


def _decode_video(data: bytes, num_frames: int = 8) -> None:
    """Decode *num_frames* equally-spaced frames from raw MP4 bytes.

    Uses torchcodec if available, otherwise falls back to PyAV.
    No-op if neither is installed.
    """
    import numpy as np
    if HAS_TORCHCODEC:
        decoder = _VideoDecoder(data)
        total = len(decoder)
        nf = min(num_frames, total)
        if nf <= 0:
            return
        indices = np.linspace(0, total - 1, nf, dtype=int).tolist()
        for i in indices:
            _ = decoder[i]
    elif HAS_AV:
        with _av.open(io.BytesIO(data)) as container:
            stream = container.streams.video[0]
            total = stream.frames or 0
            all_frames = list(container.decode(stream))
            total = len(all_frames)
            nf = min(num_frames, total)
            if nf <= 0:
                return
            indices = set(np.linspace(0, total - 1, nf, dtype=int).tolist())
            for i, frame in enumerate(all_frames):
                if i in indices:
                    _ = frame.to_ndarray(format="rgb24")


def _count_bytes(v) -> int:
    if isinstance(v, (bytes, bytearray, memoryview)):
        return len(v)
    if isinstance(v, str):
        return len(v.encode())
    if isinstance(v, list):
        return sum(_count_bytes(x) for x in v)
    return 0


# ---------------------------------------------------------------------------
# WebDataset benchmark
# ─────────────────────────────────────────────────────────────────────────
# Uses the identical IterableDataset.from_generator path as loader.py so
# that shuffling, shard assignment, and DataLoader workers all behave the
# same way they will in real training.
# ---------------------------------------------------------------------------
def bench_wds(
    num_samples: int,
    num_workers: int,
    shuffle_buffer: int,
    decode_images: bool,
    decode_video: bool = False,
    num_frames: int = 8,
    seed: int = 42,
) -> BenchResult:
    import glob
    import numpy as np
    from datasets import IterableDataset
    from torch.utils.data import DataLoader
    from llamafactory.data.loader import _iter_webdataset_single_tar

    t_init0 = time.perf_counter()

    wds_files = sorted(glob.glob(WDS_GLOB))
    # Mirror loader.py: shuffle shard order with training seed
    rng = np.random.default_rng(seed)
    wds_files = [wds_files[i] for i in rng.permutation(len(wds_files))]

    # Mirror loader.py: IterableDataset with one logical shard per TAR
    dataset = IterableDataset.from_generator(
        _iter_webdataset_single_tar,
        gen_kwargs={"wds_file": wds_files},
    )
    if shuffle_buffer > 0:
        dataset = dataset.shuffle(buffer_size=shuffle_buffer, seed=seed)

    loader = DataLoader(
        dataset,
        batch_size=None,          # samples already shaped by WDS
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=lambda x: x,  # pass dicts through unchanged
    )

    t_init1 = time.perf_counter()
    init_s = t_init1 - t_init0

    total_bytes = 0
    count = 0
    errors = 0

    t0 = time.perf_counter()
    for sample in loader:
        try:
            # "jpg" key holds a list of raw JPEG bytes (one per image in sample)
            img_list = sample.get("jpg", []) or []
            for raw in img_list:
                total_bytes += len(raw)
                if decode_images:
                    try:
                        _decode_bytes(raw)
                    except Exception as e:
                        print(f"  Image decode error: {e}")
                        errors += 1
            # "mp4" key holds a list of raw MP4 bytes (one per video clip in sample)
            vid_list = sample.get("mp4", []) or []
            for raw in vid_list:
                total_bytes += len(raw)
                if decode_video:
                    try:
                        _decode_video(raw, num_frames)
                    except Exception as e:
                        print(f"  Video decode error: {e}")
                        errors += 1
            # Count text fields too (messages, id, …)
            for k, v in sample.items():
                if k not in ("jpg", "jpeg", "png", "mp4"):
                    total_bytes += _count_bytes(v)
        except Exception as e:
            print(f"  Sample processing error: {e}")
            errors += 1
        count += 1
        if count >= num_samples:
            break
    iter_s = time.perf_counter() - t0

    dec = ("img+vid" if decode_video else "img") if decode_images else ("vid" if decode_video else "no-dec")
    shuf = f"shuf={shuffle_buffer}" if shuffle_buffer else "no-shuf"
    label = f"WDS  workers={num_workers:2d}  {shuf}  {dec}"
    return BenchResult(label, count, init_s + iter_s, init_s, iter_s, total_bytes, errors)


# ---------------------------------------------------------------------------
# LeRobot benchmark
# ─────────────────────────────────────────────────────────────────────────
# Mirrors the training pipeline exactly:
#   1. Read JSONL (map-style, text-only — same as load_lerobot_jsonl_as_dataset)
#   2. Each worker calls load_lerobot_video_frames(ref) for every lerobot:// URI
#      → LeRobotDataset opened lazily per-process (same as lerobot_bridge.py)
#   3. Returns total frame count + byte estimate per sample.
# ---------------------------------------------------------------------------

class _LeroRobotMapDataset(torch.utils.data.Dataset):
    """Map-style dataset over a JSONL of lerobot:// video references.

    Each __getitem__ call opens the LeRobot dataset lazily (per-worker cache
    via lerobot_bridge._get_lerobot_dataset, same as training) and decodes all
    frames for every ``lerobot://episode:...`` URI in the sample's ``videos``
    list.
    """

    def __init__(
        self,
        samples: list[dict],
        datasets_map: dict[str, str],
        decode_video: bool = True,
        num_frames: int = 8,
    ):
        self.samples = samples
        self.datasets_map = datasets_map
        self.decode_video = decode_video
        self.num_frames = num_frames

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, item: int) -> dict:
        import json as _json
        from llamafactory.data.lerobot_bridge import load_lerobot_video_frames

        # Set LEROBOT_DATASETS in this worker process so lerobot_bridge can
        # resolve short names ("droid", "bridge") to filesystem paths.
        # os.environ is per-process so this is safe even under fork.
        if "LEROBOT_DATASETS" not in os.environ:
            os.environ["LEROBOT_DATASETS"] = _json.dumps(self.datasets_map)

        sample = self.samples[item]
        total_frames = 0
        total_bytes = 0
        errors = 0

        for ref in sample.get("videos", []) or []:
            try:
                if self.decode_video:
                    frames = load_lerobot_video_frames(ref, num_frames=self.num_frames)  # list[PIL.Image]
                    total_frames += len(frames)
                    # Estimate bytes: H×W×3 per frame
                    if frames:
                        w, h = frames[0].size
                        total_bytes += len(frames) * h * w * 3
                    del frames
                else:
                    # Count frames without decoding: use episode metadata only
                    from llamafactory.data.lerobot_bridge import get_lerobot_video_info
                    n, _, _, _ = get_lerobot_video_info(ref, num_frames=self.num_frames)
                    total_frames += n
            except Exception as e:
                errors += 1

        return {"frames": total_frames, "bytes": total_bytes, "errors": errors}


def _lerobot_collate(batch):
    """Sum frame/byte counts from a length-1 batch (batch_size=1)."""
    return batch[0]


def bench_lerobot(
    jsonl_path: str,
    datasets_map: dict[str, str],
    num_samples: int,
    num_workers: int,
    shuffle: bool,
    decode_video: bool = True,
    num_frames: int = 8,
    seed: int = 42,
) -> BenchResult:
    """Benchmark LeRobot JSONL video loading using the same bridge as training."""
    import json as _json
    import numpy as np

    # Set LEROBOT_DATASETS in the parent **before** forking workers so they
    # inherit it.  (Workers also set it on first __getitem__ as a safety net.)
    os.environ.setdefault("LEROBOT_DATASETS", _json.dumps(datasets_map))

    t_init0 = time.perf_counter()

    with open(jsonl_path) as f:
        all_samples = [_json.loads(line) for line in f]

    rng = np.random.default_rng(seed)
    if shuffle:
        indices = rng.permutation(len(all_samples))[:num_samples]
    else:
        indices = np.arange(min(num_samples, len(all_samples)))
    selected = [all_samples[i] for i in indices]

    ds = _LeroRobotMapDataset(selected, datasets_map, decode_video=decode_video, num_frames=num_frames)

    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        collate_fn=_lerobot_collate,
        persistent_workers=num_workers > 0,
    )

    t_init1 = time.perf_counter()
    init_s = t_init1 - t_init0

    total_frames = 0
    total_bytes = 0
    count = 0
    errors = 0
    peak_pss_mb = 0.0

    with PeakRSSTracker(interval=0.1) as rss:
        t0 = time.perf_counter()
        for item in loader:
            total_frames += item["frames"]
            total_bytes += item["bytes"]
            errors += item["errors"]
            count += 1
        iter_s = time.perf_counter() - t0
        peak_pss_mb = rss.peak_mb

    frames_per_sec = total_frames / iter_s if iter_s > 0 else 0.0
    shuf = "shuffle" if shuffle else "no-shuf"
    dec = "decode" if decode_video else "no-decode"
    label = f"LeRobot workers={num_workers:2d}  {shuf}  {dec}  nf={num_frames}"
    result = BenchResult(label, count, init_s + iter_s, init_s, iter_s, total_bytes, peak_pss_mb, errors)
    # Store frames/s as a note — print it right after returning
    result._frames_per_sec = frames_per_sec  # type: ignore[attr-defined]
    result._total_frames = total_frames      # type: ignore[attr-defined]
    return result


# ---------------------------------------------------------------------------
# Lance benchmark
# ─────────────────────────────────────────────────────────────────────────
# The dataset stores only (lance_path, indices, col names) — no Arrow table,
# no open file handles.  Each DataLoader worker opens its own lance handle on
# the first __getitem__ call via _get_lance_dataset (per-process cache).
# Because the parent never opens lance, fork is safe and spawn is not needed.
# ---------------------------------------------------------------------------

def _collate_first(batch):
    """Unwrap batch-of-1. Top-level so it's picklable if spawn is ever needed."""
    return batch[0]


def _is_video_col(name: str) -> bool:
    """Heuristic: blob column names containing 'mp4' or 'video' are video columns."""
    n = name.lower()
    return "mp4" in n or "video" in n


class _LanceMapDataset(torch.utils.data.Dataset):
    """Map-style dataset that opens lance lazily inside each worker.

    Only the path string and the pre-shuffled index array are stored — nothing
    that requires pickling a large object.  On the first __getitem__ call in
    each worker process, _get_lance_dataset opens and caches the handle for
    that process.  Because the parent never opens lance before fork, no
    fork-safety issues arise and spawn is unnecessary.
    """

    def __init__(
        self,
        lance_path: str,
        indices,          # numpy int64 array, length == num_samples
        scalar_cols: list[str],
        blob_cols: list[str],
        decode_images: bool,
        decode_video: bool = False,
        num_frames: int = 8,
    ):
        self.lance_path = lance_path
        self.indices = indices
        self.scalar_cols = scalar_cols
        # Split blob columns into image vs video based on column name.
        self.image_blob_cols = [c for c in blob_cols if not _is_video_col(c)]
        self.video_blob_cols  = [c for c in blob_cols if _is_video_col(c)]
        self.decode_images = decode_images
        self.decode_video  = decode_video
        self.num_frames    = num_frames

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, item):
        # Import inside __getitem__ so the cache is per-worker-process.
        # _get_lance_dataset opens lance on first call, then reuses the handle.
        from llamafactory.data.lance_utils import _get_lance_dataset

        idx = int(self.indices[item])
        ds = _get_lance_dataset(self.lance_path)

        total_bytes = 0
        errors = 0

        # Scalar columns — cheap batch read
        if self.scalar_cols:
            row = ds.take([idx], columns=self.scalar_cols).to_pydict()
            for vals in row.values():
                total_bytes += _count_bytes(vals[0] if vals else "")

        # Image blob columns — take_blobs gives a file-like handle, no full copy until .read()
        for col in self.image_blob_cols:
            try:
                blobs = ds.take_blobs(col, indices=[idx])
                raw = blobs[0].read()
                blobs[0].close()
                total_bytes += len(raw)
                if self.decode_images:
                    try:
                        _decode_bytes(raw)
                    except Exception as e:
                        print(f"  Image decode error: {e}")
                        errors += 1
            except Exception:
                errors += 1

        # Video blob columns — MP4 bytes decoded via torchcodec
        for col in self.video_blob_cols:
            try:
                blobs = ds.take_blobs(col, indices=[idx])
                raw = blobs[0].read()
                blobs[0].close()
                total_bytes += len(raw)
                if self.decode_video:
                    try:
                        _decode_video(raw, self.num_frames)
                    except Exception as e:
                        print(f"  Video decode error: {e}")
                        errors += 1
            except Exception:
                errors += 1

        return {"bytes": total_bytes, "errors": errors}


def _cap_lance_threads(n: int) -> None:
    """Cap all thread pools that lance workers will create after fork.

    Lance-specific vars (highest priority):
      LANCE_IO_THREADS   — I/O thread pool (object store / disk reads)
      LANCE_CPU_THREADS  — compute thread pool (decoding, encoding)

    Generic fallback pools also used by lance's Rust deps:
      RAYON_NUM_THREADS      — rayon parallel iterator thread pool
      TOKIO_WORKER_THREADS   — tokio async runtime threads
      OMP_NUM_THREADS        — OpenMP (numpy / PIL / torch ops in workers)
      MKL_NUM_THREADS        — Intel MKL
      OPENBLAS_NUM_THREADS   — OpenBLAS

    All must be set in the PARENT before DataLoader forks so workers
    inherit them and create small runtimes on first use.

    Uncapped default on a 72-core machine: ~85 OS threads per worker
    Capped to 1: ~4-6 OS threads per worker
    32 workers × 85 = 2720 > ulimit -u=4096  →  SIGSEGV
    32 workers ×  6 =  192 < 4096            →  OK
    """
    if n <= 0:
        # Remove caps — workers see lance/rayon/tokio defaults
        for var in (
            "LANCE_IO_THREADS", "LANCE_CPU_THREADS",
        ):
            os.environ.pop(var, None)
        return
    ns = str(n)
    # Hard-set (not setdefault) so sweeping different values actually changes
    # the env that forked workers inherit.
    for var in (
        "LANCE_IO_THREADS", "LANCE_CPU_THREADS",
    ):
        os.environ[var] = ns


def bench_lance(
    num_samples: int,
    num_workers: int,
    shuffle: bool,
    decode_images: bool,
    decode_video: bool = False,
    num_frames: int = 8,
    lance_threads: int = 2,
    seed: int = 42,
) -> BenchResult:
    import numpy as np
    import lance as _lance
    import pyarrow as pa
    from llamafactory.data import lance_utils

    # Clear the per-process handle cache so every run starts cold.
    lance_utils._LANCE_HANDLES.clear()

    # Cap per-worker thread pools **before** DataLoader forks workers so the
    # env vars are inherited.  Lance opens its dataset lazily inside each
    # worker and will create tokio/rayon pools respecting these limits.
    if num_workers > 0:
        _cap_lance_threads(lance_threads)
        capped = lance_threads > 0
        per_worker = lance_threads * 2 + 2 if capped else 85  # ~4-6 capped, ~85 uncapped
        budget = num_workers * per_worker
        print(
            f"    [thread-cap] LANCE_IO={os.environ.get('LANCE_IO_THREADS', '?')}  "
            f"LANCE_CPU={os.environ.get('LANCE_CPU_THREADS', '?')}  "
            f"RAYON={os.environ.get('RAYON_NUM_THREADS', '?')}  "
            f"TOKIO={os.environ.get('TOKIO_WORKER_THREADS', '?')}  "
            f"OMP={os.environ.get('OMP_NUM_THREADS', '?')}  "
            f"est. threads/{num_workers}w ≈ {budget}  (ulimit -u {_ULIMIT_U})"
        )

    t_init0 = time.perf_counter()

    # Inspect schema to split scalar vs blob columns — parent never keeps the
    # handle open, so workers can fork safely.
    _tmp_ds = _lance.dataset(LANCE_PATH)
    schema = _tmp_ds.schema

    def _is_blob(field):
        if pa.types.is_binary(field.type) or pa.types.is_large_binary(field.type):
            return True
        return isinstance(field.type, pa.ExtensionType) and "blob" in getattr(field.type, "extension_name", "").lower()

    blob_cols   = [f.name for f in schema if _is_blob(f)]
    scalar_cols = [f.name for f in schema if not _is_blob(f)]
    n_rows      = _tmp_ds.count_rows()
    del _tmp_ds                    # close immediately — workers must open their own
    lance_utils._LANCE_HANDLES.clear()

    indices = np.arange(n_rows)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)
    indices = indices[:num_samples]

    ds = _LanceMapDataset(LANCE_PATH, indices, scalar_cols, blob_cols, decode_images, decode_video, num_frames)

    # Fork is safe: parent never opened lance.  No spawn overhead.
    loader = torch.utils.data.DataLoader(
        ds,
        batch_size=1,
        num_workers=num_workers,
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=_collate_first,
        persistent_workers=num_workers > 0,  # keep workers alive across potential re-use
    )

    t_init1 = time.perf_counter()
    init_s = t_init1 - t_init0

    total_bytes = 0
    count = 0
    errors = 0
    peak_rss_mb = 0.0

    with PeakRSSTracker(interval=0.1) as rss:
        t0 = time.perf_counter()
        for item in loader:
            total_bytes += item["bytes"]
            errors += item["errors"]
            count += 1
        iter_s = time.perf_counter() - t0
        peak_rss_mb = rss.peak_mb

    shuf = "shuffle" if shuffle else "no-shuf"
    access = "rand" if shuffle else "seq"
    thr = f" threads={lance_threads}" if num_workers > 0 else ""
    dec = ("img+vid" if decode_video else "img") if decode_images else ("vid" if decode_video else "no-dec")
    label = f"Lance workers={num_workers:2d}{thr}  {shuf}({access})  {dec}"
    return BenchResult(label, count, init_s + iter_s, init_s, iter_s, total_bytes, peak_rss_mb, errors)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def print_header():
    w = 130
    print()
    print("=" * w)
    print(f"{'Label':<58s}  {'Samples':>7s}  {'init(s)':>7s}  {'iter(s)':>7s}  {'samp/s':>8s}  {'MB/s':>7s}  {'peak_pss':>10s}")
    print("=" * w)


def run_all(args):
    results: list[BenchResult] = []
    print_header()

    worker_configs: list[int] = args.workers
    shuffle_configs = [False, True] if args.shuffle else [False]
    # ---- LeRobot ----
    if args.lerobot_jsonl:
        import json as _json
        datasets_map: dict[str, str] = {}
        # Priority: --lerobot-datasets CLI > LEROBOT_DATASETS env > built-in default
        if args.lerobot_datasets:
            datasets_map = _json.loads(args.lerobot_datasets)
        elif os.environ.get("LEROBOT_DATASETS"):
            datasets_map = _json.loads(os.environ["LEROBOT_DATASETS"])
        else:
            datasets_map = dict(LEROBOT_DATASETS_DEFAULT)

        print(f"\n[LeRobot  \u2014 JSONL={args.lerobot_jsonl}]")
        print(f"  datasets_map: {datasets_map}")
        for nw in worker_configs:
            for shuf in shuffle_configs:
                try:
                    r = bench_lerobot(
                        jsonl_path=args.lerobot_jsonl,
                        datasets_map=datasets_map,
                        num_samples=args.num_samples,
                        num_workers=nw,
                        shuffle=shuf,
                        decode_video=args.decode_video,
                        num_frames=args.lerobot_num_frames,
                        seed=args.seed,
                    )
                    results.append(r)
                    fps = getattr(r, "_frames_per_sec", 0.0)
                    total_f = getattr(r, "_total_frames", 0)
                    print(f"{r}  {fps:>8.1f} frames/s  {total_f} total frames")
                except Exception as e:
                    import traceback
                    print(f"  LeRobot workers={nw} shuffle={shuf} FAILED: {e}")
                    traceback.print_exc()
    # ---- WebDataset ----
    print("\n[WebDataset]")
    for nw in worker_configs:
        for shuf in shuffle_configs:
            buf = args.shuffle_buffer if shuf else 0
            try:
                r = bench_wds(
                    num_samples=args.num_samples,
                    num_workers=nw,
                    shuffle_buffer=buf,
                    decode_images=args.decode,
                    decode_video=args.decode_video,
                    num_frames=args.lerobot_num_frames,
                    seed=args.seed,
                )
                results.append(r)
                print(r)
            except Exception as e:
                print(f"  WDS workers={nw} shuffle={shuf} FAILED: {e}")

    # ---- Lance ----
    thread_configs: list[int] = args.lance_threads
    print("\n[Lance  \u2014 fork workers, lazy dataset open per worker]")
    for lt in thread_configs:
        for nw in worker_configs:
            for shuf in shuffle_configs:
                try:
                    r = bench_lance(
                        num_samples=args.num_samples,
                        num_workers=nw,
                        shuffle=shuf,
                        decode_images=args.decode,
                        decode_video=args.decode_video,
                        num_frames=args.lerobot_num_frames,
                        lance_threads=lt,
                        seed=args.seed,
                    )
                    results.append(r)
                    print(r)
                except Exception as e:
                    print(f"  Lance workers={nw} threads={lt} shuffle={shuf} FAILED: {e}")



    # ---- Summary ----
    if results:
        bs = args.train_batch_size
        w = 130
        print("\n" + "=" * w)
        print(f"SUMMARY  (sorted by iter samp/s desc — init overhead excluded)")
        print(f"         Training batch size = {bs}  →  need {bs} samp/s to sustain 1 batch/s")
        print("=" * w)
        for r in sorted(results, key=lambda x: x.samples_per_sec, reverse=True):
            bps = r.batches_per_sec(bs)
            spb = r.secs_per_batch(bs)
            bottleneck = "  *** BOTTLENECK" if spb > 0.5 else ""
            print(f"{r}  {bps:>6.2f} batch/s  {spb:>5.2f}s/batch{bottleneck}")
        print()
        print("NOTE:")
        print("  init(s)  = dataset load + index shuffle + DataLoader/worker startup (one-time cost)")
        print("  iter(s)  = pure data iteration loop (what matters for training throughput)")
        print("  Lance seq  = sequential row access  → OS read-ahead works, fast")
        print("  Lance rand = random row access       → random seeks, no read-ahead, ~10x slower")
        print("  LeRobot    = each sample = 1 full episode (100-500 frames) decoded from MP4 via lerobot_bridge")
        print("             → expect 0.5-3 samp/s (=50-1500 frames/s) on CPU; this IS the bottleneck for bs=64")
        print("             → mitigation: pre-extract frames to JPEG/lance at prep time (see prepare_lerobot_vqa.py)")
        print("  Thread-limit cause: rayon+tokio+OMP runtimes each default to nproc per worker (~85 OS threads total)")
        print(f"  This host: ulimit -u={_ULIMIT_U}, nproc={os.cpu_count()}")
        print("  --lance-threads N  hard-sets all pools (RAYON+TOKIO+OMP+MKL+OPENBLAS) in parent env before fork")
        print("  lance-threads=1→~4 OS threads/worker | =2→~6 | =0→uncapped (~85, segfaults >~48 workers here)")
        print(f"  Bottleneck threshold: >0.5s/batch (assumes GPU step ≥ 0.5s; adjust for your hardware)")
    print()


def parse_args():
    p = argparse.ArgumentParser(description="Lance vs WebDataset dataloader benchmark (uses LlamaFactory utils)")
    p.add_argument("--num-samples", type=int, default=2000,
                   help="Samples to consume per run (default: 2000)")
    p.add_argument("--workers", type=int, nargs="+", default=[2, 4, 8, 16, 32],
                   help="num_workers for DataLoader (default: 2 4 8 16 32)")
    p.add_argument("--lance-threads", type=int, nargs="+", default=[1, 2, 4],
                   help=(
                       "One or more thread-cap values to sweep (default: 1 2 4). "
                       "Each value sets RAYON/TOKIO/OMP/MKL/OPENBLAS_NUM_THREADS per worker before fork. "
                       "Without capping each worker spawns ~85 OS threads; ulimit -u=4096 means "
                       "32 workers × 85 = 2720 > 4096 → SIGSEGV. Use 0 to benchmark uncapped."
                   ))
    p.add_argument("--shuffle", action="store_true", default=True,
                   help="Also benchmark with shuffling (default: True)")
    p.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    p.add_argument("--shuffle-buffer", type=int, default=2000,
                   help="WDS shuffle buffer size (default: 2000)")
    p.add_argument("--decode", action="store_true", default=True,
                   help="PIL-decode JPEG bytes (mirrors training pipeline, default: True)")
    p.add_argument("--no-decode", dest="decode", action="store_false",
                   help="Skip PIL decode — measure raw I/O only")
    p.add_argument("--decode-video", action="store_true", default=True,
                   help="torchcodec-decode MP4 blobs (all frames); default: off (raw I/O only)")
    p.add_argument("--train-batch-size", type=int, default=64,
                   help="Training batch size used to compute batch/s and s/batch in the summary (default: 64)")
    p.add_argument("--lerobot-jsonl", type=str,
                   default=LEROBOT_JSONL,
                   help="Path to the LeRobot training JSONL (default: LEROBOT_JSONL constant). Pass empty string to skip.")
    p.add_argument("--lerobot-datasets", type=str, default="",
                   help=(
                       'JSON string mapping dataset short names to paths, e.g.: '
                       '{\"droid\":\"/path/to/droid\",\"bridge\":\"/path/to/bridge\"}. '
                       'Falls back to LEROBOT_DATASETS env var, then built-in defaults.'
                   ))
    p.add_argument("--lerobot-num-frames", type=int, default=8,
                   help="Number of frames to uniformly subsample per episode in the LeRobot benchmark (default: 8)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# Read at import time so the summary note always has the right number.
_ULIMIT_U: int = 0
try:
    import resource as _resource
    _ULIMIT_U = _resource.getrlimit(_resource.RLIMIT_NPROC)[0]
except Exception:
    pass


if __name__ == "__main__":
    args = parse_args()

    print(f"PIL available      : {HAS_PIL}")
    print(f"torchcodec avail   : {HAS_TORCHCODEC}")
    print(f"pyav avail         : {HAS_AV}")
    print(f"num_samples        : {args.num_samples}")
    print(f"workers            : {args.workers}")
    print(f"lance-threads      : {args.lance_threads}  (values to sweep; 0=uncapped)")
    print(f"  ulimit -u        : {_ULIMIT_U}  (per-user OS task limit)")
    print(f"  safe max workers (est): {_ULIMIT_U // 85} uncapped | {_ULIMIT_U // 6} capped-to-2")
    print(f"shuffle            : {args.shuffle}  (buffer={args.shuffle_buffer})")
    print(f"decode images      : {args.decode}")
    _video_backend = "torchcodec" if HAS_TORCHCODEC else ("pyav" if HAS_AV else "none")
    print(f"decode video       : {args.decode_video}  (backend={_video_backend}; all frames)")
    print(f"train batch size   : {args.train_batch_size}  (for batch/s and s/batch in summary)")
    print(f"Lance path         : {LANCE_PATH}")
    print(f"WDS path           : {WDS_DIR}")
    print(f"LeRobot JSONL      : {args.lerobot_jsonl or '(skipped)'}")
    print(f"LeRobot num_frames : {args.lerobot_num_frames}  (uniformly subsampled per episode)")

    run_all(args)
