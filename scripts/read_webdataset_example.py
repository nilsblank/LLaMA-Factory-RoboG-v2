#!/usr/bin/env python3
"""
Example script to read and inspect webdataset tar files created by prepare_lerobot_vqa.py
"""

import json
import sys
from pathlib import Path

import webdataset as wds
import cv2
import numpy as np


def read_webdataset(tar_pattern: str, max_samples: int = 5):
    """Read and display webdataset samples.
    
    Args:
        tar_pattern: Glob pattern for tar files (e.g., "/output/webdataset/*.tar")
        max_samples: Maximum number of samples to inspect
    """
    print(f"Reading webdataset from: {tar_pattern}\n")
    
    dataset = wds.WebDataset(tar_pattern)
    
    for i, sample in enumerate(dataset):
        if i >= max_samples:
            break
        
        # Extract components
        key = sample["__key__"]
        video_bytes = sample["mp4"]
        metadata = json.loads(sample["json"])
        
        print(f"Sample {i+1}:")
        print(f"  Key: {key}")
        print(f"  Video size: {len(video_bytes):,} bytes")
        print(f"  Dataset: {metadata.get('dataset', 'N/A')}")
        print(f"  Episode: {metadata.get('episode_idx', 'N/A')}")
        print(f"  Camera: {metadata.get('camera_key', 'N/A')}")
        print(f"  Conversations: {len(metadata.get('conversations', []))}")
        
        # Display conversations
        for j, conv in enumerate(metadata.get("conversations", [])):
            print(f"    Conversation {j+1}:")
            for turn in conv:
                role = turn.get("role", "unknown")
                content = turn.get("content", "")[:100]  # First 100 chars
                print(f"      {role}: {content}...")
        
        # Optional: Save first frame as image for inspection
        if i == 0:
            save_first_frame(video_bytes, f"sample_{i}_first_frame.jpg")
        
        print()


def save_first_frame(video_bytes: bytes, output_path: str):
    """Extract and save the first frame from video bytes."""
    # Write bytes to temporary file
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp.write(video_bytes)
        tmp_path = tmp.name
    
    # Read first frame
    cap = cv2.VideoCapture(tmp_path)
    ret, frame = cap.read()
    cap.release()
    
    if ret:
        cv2.imwrite(output_path, frame)
        print(f"  Saved first frame to: {output_path}")
    
    # Clean up
    Path(tmp_path).unlink(missing_ok=True)


def inspect_webdataset_structure(tar_pattern: str):
    """Inspect the structure of webdataset tar files."""
    import tarfile
    import glob
    
    tar_files = glob.glob(tar_pattern)
    if not tar_files:
        print(f"No tar files found matching: {tar_pattern}")
        return
    
    print(f"Found {len(tar_files)} tar files\n")
    
    # Inspect first tar file
    first_tar = tar_files[0]
    print(f"Inspecting: {first_tar}")
    
    with tarfile.open(first_tar, "r") as tar:
        members = tar.getmembers()
        print(f"  Total entries: {len(members)}")
        
        # Count by extension
        extensions = {}
        for member in members:
            ext = Path(member.name).suffix
            extensions[ext] = extensions.get(ext, 0) + 1
        
        print("  File types:")
        for ext, count in sorted(extensions.items()):
            print(f"    {ext or 'no extension'}: {count}")
        
        # Show first few entries
        print("\n  First 10 entries:")
        for member in members[:10]:
            print(f"    {member.name} ({member.size:,} bytes)")


def count_total_samples(tar_pattern: str) -> int:
    """Count total number of samples across all tar files."""
    dataset = wds.WebDataset(tar_pattern)
    count = sum(1 for _ in dataset)
    return count


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python read_webdataset_example.py <tar_pattern>")
        print("Example: python read_webdataset_example.py '/output/webdataset/*.tar'")
        sys.exit(1)
    
    tar_pattern = sys.argv[1]
    
    print("=" * 80)
    print("Webdataset Structure Inspection")
    print("=" * 80)
    inspect_webdataset_structure(tar_pattern)
    
    print("\n" + "=" * 80)
    print("Sample Data Inspection")
    print("=" * 80)
    read_webdataset(tar_pattern, max_samples=3)
    
    print("=" * 80)
    print("Counting total samples...")
    total = count_total_samples(tar_pattern)
    print(f"Total samples: {total}")
    print("=" * 80)
