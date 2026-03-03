#!/usr/bin/env python3
"""Benchmark prepare_lerobot_vqa.py with different thread/worker configs."""

import subprocess
import time
import os
import sys

# Small subset: 2 datasets, 10 episodes each via subsample
BASE_CMD = [
    sys.executable, "scripts/prepare_lerobot_vqa.py",
]

# We'll patch the script to only process a small subset.
# Instead, let's benchmark the core decode function directly.

import json
import numpy as np

def bench_decode_config(num_ffmpeg_threads, num_workers, n_episodes=20):
    """Benchmark episode video extraction with given config."""
    os.environ["LEROBOT_NUM_FFMPEG_THREADS"] = str(num_ffmpeg_threads)
    
    from torchcodec.decoders import VideoDecoder
    from torch.utils.data import Dataset, DataLoader
    import torch

    av1_path = "/e/home/jusers/blank4/jupiter/datasets/lerobot_3_0/DROID/droid_success/videos/observation.images.left_external/chunk-000/file_000.mp4"
    
    # Get episode boundaries from the dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset(
        repo_id="local",
        root="/e/home/jusers/blank4/jupiter/datasets/lerobot_3_0/DROID/droid_success",
    )
    
    # Collect first N episodes' frame ranges
    episodes = []
    for ep_idx in range(min(n_episodes, ds.num_episodes)):
        ep = ds.meta.episodes[ep_idx]
        cam = "observation.images.left_external"
        from_ts = ep[f"videos/{cam}/from_timestamp"]
        to_ts = ep[f"videos/{cam}/to_timestamp"]
        video_path = str(ds.root / ds.meta.get_video_file_path(ep_idx, cam))
        episodes.append((video_path, from_ts, to_ts))
    
    del ds  # free memory
    
    class EpisodeDataset(Dataset):
        def __init__(self, episodes, nft):
            self.episodes = episodes
            self.nft = nft
        def __len__(self):
            return len(self.episodes)
        def __getitem__(self, idx):
            path, from_ts, to_ts = self.episodes[idx]
            decoder = VideoDecoder(path, device="cpu", num_ffmpeg_threads=self.nft)
            fps = decoder.metadata.average_fps
            start = round(from_ts * fps)
            stop = round(to_ts * fps)
            result = decoder.get_frames_in_range(start=start, stop=stop)
            return result.data  # [N, C, H, W]
    
    dataset = EpisodeDataset(episodes, num_ffmpeg_threads)
    
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        multiprocessing_context="fork" if num_workers > 0 else None,
    )
    
    # Warmup
    for i, batch in enumerate(loader):
        if i >= 2:
            break
    
    # Benchmark
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=num_workers,
        multiprocessing_context="fork" if num_workers > 0 else None,
    )
    
    t0 = time.perf_counter()
    for batch in loader:
        pass
    elapsed = time.perf_counter() - t0
    
    return elapsed, elapsed / len(episodes)


if __name__ == "__main__":
    configs = [
        # (num_ffmpeg_threads, num_workers)
        (1,  0),
        (8,  0),
        (1,  8),
        (1, 16),
        (2,  8),
        (2, 16),
        (4,  8),
        (4, 16),
        (8,  4),
        (8,  8),
        (4, 32),
        (4, 64),
        (8, 32)
        
    ]
    
    N_EPISODES = 20
    
    print(f"{'Config':35s} {'Total':>10s} {'Per-ep':>10s} {'Throughput':>12s}")
    print("=" * 70)
    
    for nft, nw in configs:
        label = f"threads={nft:2d}, workers={nw:2d}"
        try:
            total, per_ep = bench_decode_config(nft, nw, N_EPISODES)
            eps_per_sec = N_EPISODES / total
            print(f"{label:35s} {total*1000:8.0f}ms {per_ep*1000:8.0f}ms {eps_per_sec:10.1f} ep/s")
        except Exception as e:
            print(f"{label:35s} FAILED: {e}")
    
    print("=" * 70)
