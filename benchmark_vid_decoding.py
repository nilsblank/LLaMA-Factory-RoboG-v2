"""
Video Decoder Benchmark: decord vs torchcodec (CPU+CUDA) vs PyAV vs PyNvVideoCodec
Tests sparse frame extraction (16 evenly spaced frames) across single and multi-worker.

Install:
    pip install decord torchcodec torch av
    pip install pynvvideocodec   # GPU nodes only

Usage:
    python benchmark_video_decoders.py
"""

import time
import subprocess
import os
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import Callable
import statistics

# ── Config ───────────────────────────────────────────────────────────────────
VIDEO_URL  = "https://commondatastorage.googleapis.com/gtv-videos-bucket/sample/ForBiggerBlazes.mp4"
#VIDEO_PATH = "/e/home/jusers/blank4/jupiter/datasets/lerobot_3_0/DROID/droid_success/videos/observation.images.left_external/chunk-000/file_000.mp4"
VIDEO_PATH = "./benchmark_video_120.mp4"
N_FRAMES_TO_SAMPLE = 32
N_REPEATS           = 5   # repeats per single-worker benchmark
N_VIDEOS_MULTI      = 32    # total decode jobs for multi-worker test
WORKER_COUNTS       = [1, 2, 4, 8]

# ── Download ──────────────────────────────────────────────────────────────────
def download_video():
    if os.path.exists(VIDEO_PATH):
        print(f"[ok] Video already exists: {VIDEO_PATH}")
        return
    print(f"[..] Downloading video to {VIDEO_PATH} ...")
    try:
        subprocess.run(["wget", "-q", "-O", VIDEO_PATH, VIDEO_URL], check=True, timeout=60)
    except Exception:
        subprocess.run(["curl", "-sL", "-o", VIDEO_PATH, VIDEO_URL], check=True, timeout=60)
    print(f"[ok] Downloaded ({os.path.getsize(VIDEO_PATH)/1e6:.1f} MB)")

# ── Decoder implementations ───────────────────────────────────────────────────

def decode_decord(path: str) -> np.ndarray:
    """Decord: sparse get_batch on 16 indices (CPU)."""
    from decord import VideoReader
    vr = VideoReader(path)
    idx = np.linspace(0, len(vr) - 1, N_FRAMES_TO_SAMPLE, dtype=int)
    return vr.get_batch(idx).asnumpy()

def decode_torchcodec_cpu(path: str) -> np.ndarray:
    """TorchCodec CPU: approximate seek mode — no upfront linear scan."""
    from torchcodec.decoders import VideoDecoder
    dec = VideoDecoder(path, device="cpu", seek_mode="approximate")
    total = dec.metadata.num_frames or len(dec)
    idx = np.linspace(0, total - 1, N_FRAMES_TO_SAMPLE, dtype=int).tolist()
    return dec.get_frames_at(indices=idx).data.numpy()

def decode_torchcodec_cuda(path: str) -> np.ndarray:
    """TorchCodec CUDA: NVDEC hardware decode, approximate seek mode."""
    import torch
    from torchcodec.decoders import VideoDecoder
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available")
    dec = VideoDecoder(path, device="cuda", seek_mode="approximate")
    total = dec.metadata.num_frames or len(dec)
    idx = np.linspace(0, total - 1, N_FRAMES_TO_SAMPLE, dtype=int).tolist()
    return dec.get_frames_at(indices=idx).data.cpu().numpy()

def decode_pyav(path: str) -> np.ndarray:
    """PyAV: seek to 16 evenly-spaced timestamps, decode nearest frame."""
    import av
    container = av.open(path)
    stream = container.streams.video[0]
    stream.codec_context.skip_frame = "NONREF"
    tb = float(stream.time_base)
    if stream.duration:
        duration = float(stream.duration * tb)
    elif stream.frames:
        duration = stream.frames / float(stream.average_rate)
    else:
        duration = 10.0
    timestamps = np.linspace(0, duration * 0.999, N_FRAMES_TO_SAMPLE)
    frames = []
    for ts in timestamps:
        container.seek(int(ts / tb), stream=stream)
        for frame in container.decode(stream):
            frames.append(frame.to_ndarray(format="rgb24"))
            break
    container.close()
    return np.stack(frames)

def decode_pynvvideocodec(path: str) -> np.ndarray:
    """PyNvVideoCodec: NVIDIA NVDEC via SimpleDecoder, indexed sampling.
    __getitem__ only supports int or slice — not list — so we call dec[i]
    per frame. get_batch_frames_by_index exists but its C++ binding rejects
    a plain Python list; individual int indexing is the safe path.
    """
    import PyNvVideoCodec as nvc
    dec = nvc.SimpleDecoder(path, gpu_id=0)
    total = len(dec)
    idx = np.linspace(0, total - 1, N_FRAMES_TO_SAMPLE, dtype=int)
    frames = []
    for i in idx:
        decoded = dec[int(i)]   # __getitem__ with a single int works reliably
        # DecodedFrame may expose .cpu() (torch tensor) or be array-like
        if hasattr(decoded, "cpu"):
            frames.append(decoded.cpu().numpy())
        else:
            frames.append(np.array(decoded))
    return np.stack(frames)
# ── Registry ──────────────────────────────────────────────────────────────────
BENCHMARKS = {
    "decord            CPU  sparse get_batch   ": decode_decord,
    "torchcodec        CPU  get_frames_at      ": decode_torchcodec_cpu,
    "torchcodec        CUDA NVDEC get_frames_at": decode_torchcodec_cuda,
    "PyAV              CPU  seek+decode 16 pts ": decode_pyav,
    "PyNvVideoCodec    CUDA NVDEC SimpleDecoder": decode_pynvvideocodec,
}

# ── Benchmarking helpers ──────────────────────────────────────────────────────
def bench_single(name: str, fn: Callable, path: str) -> dict:
    try:
        fn(path)  # warmup
    except Exception as e:
        return {"name": name, "error": str(e)}
    times = []
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        fn(path)
        times.append(time.perf_counter() - t0)
    return {
        "name":      name,
        "mean_ms":   statistics.mean(times) * 1000,
        "median_ms": statistics.median(times) * 1000,
        "stdev_ms":  statistics.stdev(times) * 1000,
        "min_ms":    min(times) * 1000,
        "max_ms":    max(times) * 1000,
    }

def bench_multi(fn: Callable, path: str, n_workers: int) -> dict:
    paths = [path] * N_VIDEOS_MULTI
    try:
        t0 = time.perf_counter()
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            list(pool.map(fn, paths))
        elapsed = time.perf_counter() - t0
        return {"workers": n_workers, "throughput": N_VIDEOS_MULTI / elapsed}
    except Exception as e:
        return {"workers": n_workers, "error": str(e)}

# ── Formatting ────────────────────────────────────────────────────────────────
W   = 78
COL = 46

def section(title):
    print(f"\n{'=' * W}")
    print(f"  {title}")
    print(f"{'=' * W}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    download_video()

    try:
        from decord import VideoReader
        vr = VideoReader(VIDEO_PATH)
        h, w = vr[0].shape[:2]
        print(f"\n[i] {len(vr)} frames  |  {w}x{h}  |  sampling {N_FRAMES_TO_SAMPLE} frames evenly")
    except Exception:
        print("[i] Could not probe video (decord not installed)")

    # ── Single-worker latency ─────────────────────────────────────────────────
    section(f"SINGLE-WORKER LATENCY  ({N_REPEATS} runs each, 1 video)")
    print(f"  {'Decoder':<{COL}} {'mean':>8}  {'median':>8}  {'std':>6}  range")
    print(f"  {'-' * (W - 2)}")

    results = {}
    for name, fn in BENCHMARKS.items():
        r = bench_single(name, fn, VIDEO_PATH)
        results[name] = r
        if "error" in r:
            short_err = r["error"][:45]
            print(f"  {name:<{COL}} SKIP  ({short_err})")
        else:
            print(f"  {name:<{COL}}"
                  f" {r['mean_ms']:>7.1f}ms"
                  f"  {r['median_ms']:>7.1f}ms"
                  f"  {r['stdev_ms']:>5.1f}ms"
                  f"  [{r['min_ms']:.1f}-{r['max_ms']:.1f}]")

    # ── Multi-worker throughput ───────────────────────────────────────────────
    section(f"MULTI-WORKER THROUGHPUT  ({N_VIDEOS_MULTI} videos, ThreadPoolExecutor)")
    hw = "  ".join(f"{w:>4}w" for w in WORKER_COUNTS)
    print(f"  {'Decoder':<{COL}}  {hw}   (vids/s)")
    print(f"  {'-' * (W - 2)}")

    for name, fn in BENCHMARKS.items():
        if "error" in results.get(name, {}):
            continue
        row = f"  {name:<{COL}}"
        for w in WORKER_COUNTS:
            r = bench_multi(fn, VIDEO_PATH, w)
            if "error" in r:
                row += f"   ERR"
            else:
                row += f"  {r['throughput']:>4.1f}"
        print(row)

    # ── Summary ───────────────────────────────────────────────────────────────
    section("SUMMARY")
    valid = {k: v for k, v in results.items() if "error" not in v}
    if valid:
        fastest = min(valid.items(), key=lambda x: x[1]["mean_ms"])
        slowest = max(valid.items(), key=lambda x: x[1]["mean_ms"])
        speedup = slowest[1]["mean_ms"] / fastest[1]["mean_ms"]
        print(f"  Fastest : {fastest[0].strip():<52} {fastest[1]['mean_ms']:.1f}ms")
        print(f"  Slowest : {slowest[0].strip():<52} {slowest[1]['mean_ms']:.1f}ms")
        print(f"  Speedup : {speedup:.1f}x between fastest and slowest")
    skipped = [k for k, v in results.items() if "error" in v]
    if skipped:
        print(f"\n  Skipped (lib not installed or no GPU):")
        for k in skipped:
            print(f"    - {k.strip()}")
    print()

if __name__ == "__main__":
    main()
