# Save this as: /e/home/jusers/blank4/jupiter/blank4/code/LLaMA-Factory-RoboG-v2/bench_decode.py

import time
import torch
from torch.utils.data import Dataset, DataLoader
from torchcodec.decoders import VideoDecoder

AV1_PATH = "/e/home/jusers/blank4/jupiter/datasets/lerobot_3_0/DROID/droid_success/videos/observation.images.left_external/chunk-000/file_000.mp4"

class TestDataset(Dataset):
    def __init__(self, n=16, device="cuda", num_ffmpeg_threads=1):
        self.n = n
        self.device = device
        self.num_ffmpeg_threads = num_ffmpeg_threads

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        decoder = VideoDecoder(
            AV1_PATH,
            device=self.device,
            num_ffmpeg_threads=self.num_ffmpeg_threads,
        )
        frames = decoder.get_frames_in_range(start=341, stop=465)
        return frames.data.cpu() if self.device == "cuda" else frames.data


def bench(label, device, num_workers, ctx, num_ffmpeg_threads=1):
    try:
        ds = TestDataset(n=16, device=device, num_ffmpeg_threads=num_ffmpeg_threads)
        dl = DataLoader(
            ds, batch_size=1, num_workers=num_workers,
            multiprocessing_context=ctx,
        )
        # Warmup
        for batch in dl:
            break

        dl = DataLoader(
            ds, batch_size=1, num_workers=num_workers,
            multiprocessing_context=ctx,
        )
        t0 = time.perf_counter()
        for batch in dl:
            pass
        elapsed = time.perf_counter() - t0
        print(f"{label:50s}  {elapsed*1000:7.0f}ms total  {elapsed/16*1000:5.0f}ms/sample")
    except Exception as e:
        print(f"{label:50s}  FAILED: {e}")


if __name__ == "__main__":
    configs = [
        # (label, device, num_workers, multiprocessing_context, num_ffmpeg_threads)
        ("GPU, 0 workers",                    "cuda", 0, None,    1),
        ("GPU, 2 workers, spawn",             "cuda", 2, "spawn", 1),
        ("GPU, 4 workers, spawn",             "cuda", 4, "spawn", 1),
        ("CPU 1t, 0 workers",                 "cpu",  0, None,    1),
        ("CPU 8t, 0 workers",                 "cpu",  0, None,    8),
        ("CPU 8t, 2 workers, fork",           "cpu",  2, "fork",  8),
        ("CPU 8t, 4 workers, fork",           "cpu",  4, "fork",  8),
        ("CPU 8t, 8 workers, fork",           "cpu",  8, "fork",  8),
        ("CPU 4t, 8 workers, fork",           "cpu",  8, "fork",  4),
        ("CPU 2t, 16 workers, fork",          "cpu", 16, "fork",  2),
        ("CPU 1t, 32 workers, fork",          "cpu", 32, "fork",  1),
        ("CPU 1t, 48 workers, fork",          "cpu", 48, "fork",  1),
        ("CPU 1t, 64 workers, fork",          "cpu", 64, "fork",  1),
        ("CPU 2t, 32 workers, fork",          "cpu", 32, "fork",  2),
        ("CPU 2t, 24 workers, fork",          "cpu", 24, "fork",  2),
        ("CPU 4t, 16 workers, fork",          "cpu", 16, "fork",  4),
        ]

    print(f"{'='*80}")
    print(f"Decoding 16 samples of 124 frames @ 720x1280 AV1")
    print(f"GPU: {torch.cuda.get_device_name()}")
    print(f"CPU cores: {torch.get_num_threads()} torch / {__import__('os').cpu_count()} os")
    print(f"{'='*80}")

    for label, device, nw, ctx, nft in configs:
        bench(label, device, nw, ctx, nft)

    print(f"{'='*80}")
