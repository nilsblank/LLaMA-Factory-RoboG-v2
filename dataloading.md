# Data Loading Architecture & Debugging Guide

## Table of Contents
1. [End-to-End Data Flow](#end-to-end-data-flow)
2. [Dataset Source Types](#dataset-source-types)
3. [Multi-Dataset Merging](#multi-dataset-merging)
4. [DataLoader & Worker Architecture](#dataloader--worker-architecture)
5. [Multimodal Decoding Pipeline](#multimodal-decoding-pipeline)
6. [Distributed Training (Multi-GPU)](#distributed-training-multi-gpu)
7. [Known Issues & Segfault Sources](#known-issues--segfault-sources)
8. [Efficiency Bottlenecks](#efficiency-bottlenecks)
9. [Key File Reference](#key-file-reference)

---

## End-to-End Data Flow

```
YAML config (dataset: eo1m_lance, lerobot_vqa_bridge_droid)
    |
    v
get_dataset()                          [loader.py:714-879]
    |
    +-- _get_merged_dataset()          [loader.py:597-619]
    |       |
    |       +-- _load_single_dataset() x N    [loader.py:347-594]
    |       |       |
    |       |       +-- Lance  -> map-style Dataset (URI strings)
    |       |       +-- LeRobot JSONL -> map-style Dataset (URI strings)
    |       |       +-- WebDataset -> IterableDataset (raw bytes)
    |       |       +-- Standard JSONL -> map-style Dataset
    |       |
    |       +-- merge_dataset()        [data_utils.py:162-216]
    |               |
    |               +-- _promote_to_iterable() if mixed types
    |               +-- concat or interleave
    |
    +-- split_dataset()                [data_utils.py:219-265]
    |       +-- .take(N) / .skip(N) for IterableDataset
    |       +-- .train_test_split() for map-style
    |
    +-- _get_preprocessed_dataset()    [loader.py:662-711]
            |
            +-- dataset.map(preprocess_dataset, ...)
            |       +-- mm_plugin.process_messages()  [NO pixel decode]
            |       +-- tokenize messages
            |       +-- attach images/videos as references (URIs, bytes, paths)
            |
            v
    DatasetModule { train_dataset, eval_dataset }
            |
            v
    HF Trainer.get_train_dataloader()  [transformers Trainer._get_dataloader]
            |
            +-- DataLoader(num_workers=N, collate_fn=collator)
            |
            v
    [Worker Process] collator.__call__()     [collator.py:108-242]
            |
            +-- mm_plugin.get_mm_inputs()    [mm_plugin.py:486-511]
            |       |
            |       +-- _regularize_images()  -> resolve lance://, lerobot:// URIs
            |       +-- _regularize_videos()  -> decode via PyAV or lerobot bridge
            |       +-- image_processor()     -> pixel_values tensors
            |
            +-- DataCollatorForSeq2Seq       -> pad input_ids, attention_mask
            +-- get_rope_index()             -> position_ids (Qwen MRope)
            |
            v
    Batch dict -> model.forward()
```

---

## Dataset Source Types

### Your Current Config (`qwen3vl_lerobot_vqa.yaml`)

```yaml
dataset: eo1m_lance, lerobot_vqa_bridge_droid
```

| Dataset | Source Type | Loaded As | Media Columns | Storage |
|---------|-----------|-----------|---------------|---------|
| `eo1m_lance` | `lance_files` | **map-style Dataset** | `image_blob` (lance blob v2) | `lance://path#col#row` URI strings |
| `lerobot_vqa_bridge_droid` | `file_name` (JSONL) | **map-style Dataset** | `videos` | file paths / raw content |

**Critical observation:** `lerobot_vqa_bridge_droid` uses `file_name` (not `lerobot_files`), so it loads as a **standard JSONL dataset** through HF `load_dataset()`, not the LeRobot bridge. The `lerobot://` URI mechanism is NOT engaged for this dataset.

### Lance Dataset Loading (`loader.py:538-555`)

1. Opens lance file with `lance.dataset(path)`
2. Partitions columns: scalar (text/int) vs binary (images/videos)
3. **Replaces binary columns with `lance://path#col#row` URI strings** (no bytes in Arrow)
4. Returns map-style HF `Dataset`
5. URI resolution deferred to collation via `resolve_lance_uri()` (`lance_utils.py:60-74`)

### LeRobot Dataset Loading (`loader.py:485-537`)

1. Loads JSONL as map-style Dataset (just text parsing)
2. Shuffles globally (true random access on map-style)
3. **Line 523: `to_iterable_dataset()` is COMMENTED OUT** -> stays map-style
4. Applies `.map(transform_fn, num_proc=preprocessing_num_workers)` to generate `lerobot://` URIs
5. URI resolution deferred to collation via `load_lerobot_frame()` / `load_lerobot_video_frames()`

### WebDataset Loading (`loader.py:432-484`)

1. Creates `IterableDataset.from_generator()` with one logical shard per TAR file
2. Images kept as raw JPEG bytes (~25KB) through entire pipeline
3. Shuffle buffer applied (`buffer_size=16384` default)

### Standard JSONL/JSON (`loader.py:556-569`)

1. HF `load_dataset()` with optional `streaming=True`
2. If `streaming=True` and `load_from == "file"`: converts to IterableDataset

---

## Multi-Dataset Merging

### Flow (`data_utils.py:162-216`)

```python
# 1. Promote types if mixed
all_datasets = _promote_to_iterable(all_datasets)  # converts map->iterable if ANY is iterable

# 2. Check schema compatibility
compatible = _features_compatible(all_datasets)

# 3. Merge
if mix_strategy == "concat":
    if compatible: concatenate_datasets(all_datasets)    # HF native
    else: _concatenate_iterable_featureless(all_datasets) # skip schema check
elif mix_strategy.startswith("interleave"):
    ...  # similar pattern
```

### Type Promotion Rules (`_promote_to_iterable`, `data_utils.py:147-159`)

- If **all map-style**: stays map-style -> `concatenate_datasets` or `interleave_datasets`
- If **all IterableDataset**: stays iterable
- If **mixed**: ALL promoted to IterableDataset via `.to_iterable_dataset()`

### Schema Compatibility

When datasets have different column schemas (e.g., one has `images`, another has `videos`):
- `_features_compatible()` calls HF's `_check_if_features_can_be_aligned()`
- If compatible: HF fills missing columns with None
- If incompatible: featureless merge (sets `info.features = None`)
- **BUG RISK**: `_concatenate_iterable_featureless()` and `_interleave_iterable_featureless()` access `.ex_iterable` which is IterableDataset-only. If both datasets are map-style AND incompatible, this will crash.

### Your Config's Merge Behavior

Both `eo1m_lance` and `lerobot_vqa_bridge_droid` are **map-style**. Default `mix_strategy=concat`:
- No type promotion (both map-style)
- Schema check: one has `images`, other has `videos` -> depends on whether HF can align
- If compatible: `concatenate_datasets()` fills missing columns with None
- Result: single map-style Dataset with both `images` and `videos` columns

---

## DataLoader & Worker Architecture

### HF Trainer's DataLoader Creation (`transformers.Trainer._get_dataloader`)

```python
dataloader_params = {
    "batch_size": batch_size,
    "collate_fn": data_collator,
    "num_workers": self.args.dataloader_num_workers,       # from YAML
    "pin_memory": self.args.dataloader_pin_memory,
    "persistent_workers": self.args.dataloader_persistent_workers,  # from YAML
}

if not isinstance(dataset, IterableDataset):
    dataloader_params["sampler"] = sampler_fn(dataset)     # DistributedSampler for DDP
    dataloader_params["drop_last"] = True
    dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor  # from YAML
    if is_training:
        dataloader_params["worker_init_fn"] = seed_worker  # JUST seeds RNG, NOT lerobot_worker_init_fn!

# For IterableDataset: NO sampler, NO prefetch_factor, NO worker_init_fn, NO drop_last
```

### CRITICAL: `worker_init_fn` Gap

The HF Trainer uses `seed_worker` (from transformers) which **only seeds random generators**. It does NOT call `lerobot_worker_init_fn`. The `lerobot_worker_init_fn` is only used in the **custom slot pretrain DataLoader** (`workflow.py:607-619`), not the main training loop.

**Impact for `lerobot://` URIs:**
- Fork-based workers (Linux default): Inherit parent's `_dataset_cache` via COW -> usually works
- Spawn-based workers: Start with empty cache -> lazy initialization on first access via auto-trigger (`lerobot_bridge.py:290-294`)
- **Risk**: First batch in each worker triggers full dataset preloading, causing a burst of I/O

**Impact for `lance://` URIs:**
- No `worker_init_fn` needed. Lance handles opened lazily per-process on first `resolve_lance_uri()` call.
- Per-process cache (`_LANCE_HANDLES`) is thread-safe and empty after fork.

### Worker Process Lifecycle

```
Main Process:
  load datasets -> tokenize -> create DataLoader
      |
      fork() x num_workers
      |
Worker N:
  [inherited] tokenized Arrow table (COW pages)
  [empty] _LANCE_HANDLES = {}
  [empty] _dataset_cache = {}
      |
  for batch in assigned_indices:
      fetch samples from Arrow table (index-based for map-style)
      -> collator.__call__(samples)
          -> mm_plugin.get_mm_inputs()
              -> _regularize_images()
                  -> resolve_lance_uri("lance://...")  # opens lance handle on first call
                  -> Image.open(BytesIO(raw_bytes))
              -> _regularize_videos()
                  -> av.open(BytesIO(raw_bytes))       # PyAV decode
                  -> OR load_lerobot_video_frames()     # if lerobot:// URI
          -> image_processor(images, return_tensors="pt")
          -> pad input_ids, create attention_mask
      return batch_tensors  # sent back to main via mp.Queue
```

### IterableDataset Worker Sharding

For IterableDataset with `num_workers=N` and `world_size=W`:
- Accelerate calls `dataset.shard(num_shards=W, index=rank)` for DDP distribution
- PyTorch DataLoader further shards per worker via `worker_info.id` / `worker_info.num_workers`
- Total effective shards = `W * N`
- Each worker gets `total_samples / (W * N)` samples

For map-style Dataset:
- `DistributedSampler` handles DDP (shuffled per-epoch with different seeds)
- DataLoader workers fetch by index from the same Arrow table (COW shared)

---

## Multimodal Decoding Pipeline

### Two-Phase Processing

**Phase 1: Message Processing (main process, during `.map()`)**
- `mm_plugin.process_messages()` (`mm_plugin.py:2069+`)
- **NO pixel decode** - only reads headers for size computation
- `_get_image_size_no_decode()`: PIL lazy open (header only) or lance metadata
- `_get_video_info_no_decode()`: av.open() probe (no frame decode)
- Expands vision token placeholders in message text
- Output: tokenized input_ids + media references (URIs/paths/bytes)

**Phase 2: MM Inputs (worker process, during collation)**
- `mm_plugin.get_mm_inputs()` (`mm_plugin.py:486-511`)
- **Actual pixel decode happens here**
- `_regularize_images()`: Resolve URIs -> PIL Image -> image_processor -> tensors
- `_regularize_videos()`: Resolve URIs -> PyAV decode -> PIL frames -> image_processor -> tensors

### URI Resolution at Collation Time

| URI Scheme | Resolution Function | What Happens |
|-----------|-------------------|-------------|
| `lance://path#col#row` | `resolve_lance_uri()` | Opens lance handle (cached), reads blob bytes |
| `lerobot://<idx>` | `load_lerobot_frame()` | Opens LeRobot dataset (cached), decodes frame to PIL |
| `lerobot://episode:<ep>::<cam>` | `load_lerobot_video_frames()` | Batch decode via `_query_videos()` |
| File path (string) | `Image.open()` / `av.open()` | Direct filesystem access |
| Raw bytes | `Image.open(BytesIO())` / `av.open(BytesIO())` | In-memory decode |

---

## Distributed Training (Multi-GPU)

### DDP Data Distribution

```
                    Main Process (rank 0)               Main Process (rank 1)
                         |                                    |
                    DistributedSampler                   DistributedSampler
                    (shard 0 of W)                       (shard 1 of W)
                         |                                    |
              +----------+----------+              +----------+----------+
              |          |          |              |          |          |
           Worker0   Worker1   Worker2         Worker0   Worker1   Worker2
              |          |          |              |          |          |
           collate    collate    collate        collate    collate    collate
              |          |          |              |          |          |
           batch0     batch1    batch2          batch3     batch4    batch5
```

### Accelerate Integration

- `dispatch_batches=False` set automatically for IterableDataset (`loader.py:830-842`)
  - Each rank fetches its own batch independently
  - Required for variable-length VLM batches (can't `torch.cat` across different pixel_values shapes)
- For map-style: default Accelerate behavior (DistributedSampler wrapping)

### FSDP/DeepSpeed Considerations

- Collator creates **fake placeholder batches** when a batch has no media but model expects it (`collator.py:123-154`)
- Prevents hanging in ZeRO-3/FSDP all-gather when some ranks have media and others don't

---

## Known Issues & Segfault Sources

### 1. Lance `take_blobs()` Concurrent Access (HIGH RISK)

**Problem:** Multiple DataLoader workers simultaneously call `_get_lance_dataset()` and `take_blobs()` on the same Lance file. While the handle is cached per-process, Lance's internal file I/O may not be safe under concurrent access from multiple forked processes sharing the same underlying file descriptor.

**Location:** `lance_utils.py:60-74`

**Symptoms:** Segfault during collation, especially with `num_workers > 1`

**Mitigation:**
- Each worker gets its own handle (cache is empty after fork)
- But the underlying OS file descriptors may be inherited and conflicting

### 2. PyAV / FFmpeg in Forked Workers (HIGH RISK)

**Problem:** PyAV (which wraps FFmpeg) is not guaranteed fork-safe. FFmpeg initializes global state (codec registries, hardware contexts) that can become corrupted after `fork()`.

**Location:** `mm_plugin.py:303-343` (video decode in `_regularize_videos`)

**Symptoms:** Segfault during video decode in worker, often intermittent

**Mitigation:** PyAV is opened fresh per-call (`av.open()` -> decode -> `container.close()`). But FFmpeg's global `avcodec_register_all()` state survives fork.

### 3. LeRobot torchcodec VideoDecoderCache After Fork (MEDIUM RISK)

**Problem:** If the main process loads any LeRobot frames before forking workers (e.g., during `.map()` preprocessing), the global `VideoDecoderCache` state is inherited by workers. torchcodec's FFmpeg state may be corrupted.

**Location:** `lerobot_bridge.py:70-108`

**Symptoms:** Segfault when first LeRobot frame is accessed in worker

**Mitigation:** `_clear_video_decoder_cache()` is called after every decode. But inherited cache from parent may already be corrupted.

### 4. PIL Image Lazy File Handles Across Fork (LOW-MEDIUM RISK)

**Problem:** PIL Image objects use lazy loading. If an Image is opened in the main process but not fully loaded, the file handle is inherited by forked workers. Concurrent seeks on the same fd cause corruption.

**Location:** `mm_plugin.py:256-287` (`_regularize_images`)

**Mitigation:** In the current design, images are stored as URIs/bytes in Arrow, not as PIL Image objects. Resolution happens in workers. Risk is low unless `process_messages()` inadvertently triggers full image loads.

### 5. Arrow Memory-Mapped Files After Fork (MEDIUM RISK)

**Problem:** HF Datasets uses Arrow memory-mapped files. When `num_proc > 1` for `.map()`, child processes inherit mmap'd regions. The main process then forks again for DataLoader workers. Multiple levels of forking with mmap'd Arrow tables can cause:
- Segfaults from concurrent mmap access
- Memory corruption if Arrow files are modified

**Location:** `loader.py:685-699` (`.map()` with `num_proc`)

**Symptoms:** Segfault during data iteration, corrupted samples

**Mitigation:** Use `num_proc=1` or process in main thread to avoid nested forks.

### 6. `_concatenate_iterable_featureless` on Map-Style Datasets (BUG)

**Problem:** If two map-style datasets have incompatible schemas, the merge code calls `_concatenate_iterable_featureless()` which accesses `._ex_iterable` (IterableDataset-only attribute). This crashes with `AttributeError`.

**Location:** `data_utils.py:180-188`

**Flow:** `_promote_to_iterable()` only promotes when types are mixed. If both are map-style but have incompatible features, no promotion happens, but the incompatible path calls iterable-only code.

---

## Efficiency Bottlenecks

### 1. `preprocessing_num_workers: 1` for LeRobot Transform

**Your config:** `preprocessing_num_workers: 1`

When `num_proc=1`, HF `.map()` runs in the main process (no multiprocessing). The LeRobot transform at `loader.py:531-537` generates URIs sequentially for all samples. Since this is just string manipulation (no I/O), it's fast but could be parallelized for large datasets.

### 2. Lance Blob Fetching is Synchronous per Worker

Each worker fetches lance blobs one-at-a-time via `take_blobs()`. No batch prefetching or async I/O. With `num_workers=16`, this creates 16 independent random-access streams to the same lance file.

**Impact:** High IOPS on storage, potential I/O contention on NFS/network filesystems.

### 3. Video Decoding Overhead in Collator

All video decoding (PyAV) happens synchronously in the collator, inside DataLoader workers. For video-heavy datasets, this becomes the bottleneck:
- Each frame: ~5-20ms decode time
- Per sample with 32 frames: ~160-640ms
- With `per_device_train_batch_size: 8`: ~1.3-5.1s per batch for video decode alone

### 4. DataLoader Prefetch Factor with IterableDataset

**HF Trainer does NOT set `prefetch_factor` for IterableDataset** (only for map-style). Your `dataloader_prefetch_factor: 4` is ignored for IterableDataset paths. PyTorch defaults to `prefetch_factor=2`.

### 5. Worker Init I/O Burst

Without `lerobot_worker_init_fn` in the main Trainer loop, each worker lazily initializes on first batch:
- First lance blob access: opens lance file + reads schema + fetches blob
- First lerobot access: loads full LeRobot dataset + video decoder init
- All `num_workers` workers do this simultaneously -> I/O storm

### 6. `max_samples: 2048` Truncation

With `max_samples: 2048` and `overwrite_cache: true`, only 2048 samples are used from each dataset before merging. This happens at `loader.py:590-592` for map-style datasets. The Arrow table is already fully loaded before truncation.

### 7. `persistent_workers: true` Memory Footprint

Each persistent worker keeps:
- Lance handles open (file descriptors + mmap pages)
- Video decoder cache (up to 4 decoders x ~50MB each)
- Arrow table mmap references
- Python interpreter overhead (~50-100MB per worker)

With `dataloader_num_workers: 16`: **16 workers x ~300MB = ~4.8GB** additional memory per GPU process.

---

## Configuration Analysis for `qwen3vl_lerobot_vqa.yaml`

```yaml
dataset: eo1m_lance, lerobot_vqa_bridge_droid
preprocessing_num_workers: 1        # single-threaded preprocessing
dataloader_num_workers: 16          # 16 DataLoader workers per GPU
dataloader_prefetch_factor: 4       # only used for map-style datasets
dataloader_persistent_workers: true # workers stay alive across epochs
max_samples: 2048                   # truncate to 2048 per dataset
val_size: 512                       # split 512 samples for eval
```

### Specific Risks for This Config

1. **16 workers x N GPUs** all doing lance blob I/O simultaneously -> storage contention
2. **`lerobot_vqa_bridge_droid` is standard JSONL** (not LeRobot bridge) -> videos loaded as file paths, decoded via PyAV in workers
3. **val_size: 512 on merged map-style** -> `train_test_split()` on concatenated dataset
4. **`persistent_workers=true` + 16 workers** -> high baseline memory per GPU
5. **No `worker_init_fn=lerobot_worker_init_fn`** in HF Trainer path (only in slot pretrain)

### Recommended Changes for Stability

```yaml
# Reduce worker count to avoid I/O contention and memory pressure
dataloader_num_workers: 4           # was 16
dataloader_persistent_workers: true # keep for cache reuse

# Increase preprocessing workers for faster tokenization
preprocessing_num_workers: 8        # was 1

# Consider adding to avoid fork issues with video decode:
# mp_start_method: spawn  # (requires worker_init_fn override)
```

---

## Key File Reference

| File | Lines | Purpose |
|------|-------|---------|
| `src/llamafactory/data/loader.py` | 347-594 | `_load_single_dataset()` - per-source loading |
| `src/llamafactory/data/loader.py` | 597-619 | `_get_merged_dataset()` - multi-dataset merge |
| `src/llamafactory/data/loader.py` | 662-711 | `_get_preprocessed_dataset()` - tokenization |
| `src/llamafactory/data/loader.py` | 714-879 | `get_dataset()` - main entry point |
| `src/llamafactory/data/data_utils.py` | 56-77 | `_features_compatible()` - schema check |
| `src/llamafactory/data/data_utils.py` | 147-159 | `_promote_to_iterable()` - type promotion |
| `src/llamafactory/data/data_utils.py` | 162-216 | `merge_dataset()` - concat/interleave |
| `src/llamafactory/data/data_utils.py` | 219-265 | `split_dataset()` - train/eval split |
| `src/llamafactory/data/lance_utils.py` | 30-47 | Per-process Lance handle cache |
| `src/llamafactory/data/lance_utils.py` | 60-74 | `resolve_lance_uri()` - blob fetch |
| `src/llamafactory/data/lerobot_bridge.py` | 66-68 | Module-level dataset cache + lock |
| `src/llamafactory/data/lerobot_bridge.py` | 70-108 | Video decoder cache cleanup |
| `src/llamafactory/data/lerobot_bridge.py` | 246-296 | `_get_lerobot_dataset()` - lazy cache |
| `src/llamafactory/data/lerobot_bridge.py` | 340-348 | `lerobot_worker_init_fn()` - worker init |
| `src/llamafactory/data/mm_plugin.py` | 256-287 | `_regularize_images()` - URI resolution |
| `src/llamafactory/data/mm_plugin.py` | 289-348 | `_regularize_videos()` - video decode |
| `src/llamafactory/data/mm_plugin.py` | 486-511 | `get_mm_inputs()` - pixel processing |
| `src/llamafactory/data/collator.py` | 85-242 | `MultiModalDataCollatorForSeq2Seq` |
| `src/llamafactory/train/sft/trainer.py` | 149-153 | Custom `_get_train_sampler()` |
| `src/llamafactory/train/sft/workflow.py` | 605-636 | Custom slot pretrain DataLoader |
| `transformers.Trainer._get_dataloader` | (external) | Default DataLoader creation |
| `data/dataset_info.json` | 1527-1540 | `lerobot_vqa_bridge_droid` config |
| `data/dataset_info.json` | 1588-1596 | `eo1m_lance` config |

---

## HF Trainer DataLoader Internals (Critical)

The HF `Trainer._get_dataloader()` creates DataLoaders with these key behaviors:

### For Map-Style Datasets
- `sampler = DistributedSampler(dataset)` (shuffled, per-rank)
- `worker_init_fn = seed_worker` (ONLY seeds RNG, not lerobot/lance init)
- `prefetch_factor` from config (honored)
- `drop_last = True`

### For IterableDataset
- **No sampler** (IterableDataset handles its own sharding)
- **No `worker_init_fn`** (workers start cold)
- **No `prefetch_factor`** (PyTorch default = 2)
- **No `drop_last`**
- Accelerate wraps with `dispatch_batches=False`

### The `worker_init_fn` Gap

The `lerobot_worker_init_fn` is **only** passed to:
1. Custom slot pretrain DataLoader (`workflow.py:619`)
2. v1 StatefulDataLoader (`v1/core/utils/batching.py:145`)

It is **NOT** passed to the standard HF Trainer DataLoader. LeRobot/Lance datasets in the main training loop rely on:
- Fork-based COW inheritance (Linux) - parent's caches inherited
- Lazy first-access initialization - each worker opens handles on demand
- Auto-trigger mechanism (`lerobot_bridge.py:290-294`) - first `lerobot://` ref triggers full preload
