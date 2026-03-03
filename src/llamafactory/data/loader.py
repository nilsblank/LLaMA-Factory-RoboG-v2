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

import glob as _glob_module
import json as _json_module
import math
import os
from typing import TYPE_CHECKING, Iterator, Literal, Optional, Union

import numpy as np
from datasets import Dataset, DatasetDict, IterableDataset, load_dataset, load_from_disk

from ..extras import logging
from ..extras.constants import FILEEXT2TYPE
from ..extras.misc import check_version, has_tokenized_data
from .converter import align_dataset
from .data_utils import get_dataset_module, merge_dataset, read_cloud_json, split_dataset
from .parser import get_dataset_list
from .processor import (
    FeedbackDatasetProcessor,
    PackedSupervisedDatasetProcessor,
    PairwiseDatasetProcessor,
    PretrainDatasetProcessor,
    SupervisedDatasetProcessor,
    UnsupervisedDatasetProcessor,
)


if TYPE_CHECKING:
    from datasets import Dataset, IterableDataset
    from transformers import PreTrainedTokenizer, ProcessorMixin, Seq2SeqTrainingArguments

    from ..hparams import DataArguments, ModelArguments
    from .data_utils import DatasetModule
    from .parser import DatasetAttr
    from .processor import DatasetProcessor
    from .template import Template


logger = logging.get_logger(__name__)

# ---------------------------------------------------------------------------
# WebDataset helpers
# ---------------------------------------------------------------------------

# File extensions that carry raw media bytes inside a WebDataset TAR shard.
_WDS_MEDIA_EXTENSIONS: frozenset[str] = frozenset(
    {
        # images
        "jpg",
        "jpeg",
        "png",
        "gif",
        "bmp",
        "tiff",
        "webp",
        "npy",
        # video
        "mp4",
        "mov",
        "avi",
        "mkv",
        "webm",
        # audio
        "wav",
        "mp3",
        "flac",
        "m4a",
        "ogg",
    }
)

# Internal WebDataset columns that should not be forwarded downstream.
_WDS_SKIP_COLUMNS: frozenset[str] = frozenset({"__key__", "__url__"})


def _iter_webdataset_tars(wds_files: list[str]) -> Iterator[dict]:
    """Iterate WebDataset TAR shards using the ``webdataset`` package.

    Why not ``load_dataset("webdataset", ...)`` from HF datasets?
    ---------------------------------------------------------------
    HF's implementation infers a **fixed Arrow schema** from the first
    ``NUM_EXAMPLES_FOR_FEATURES_INFERENCE`` (= 5) examples of the first shard
    only.  When shards mix single-image and multi-image samples the extra image
    columns (e.g. ``1.jpg``) are absent from those first examples and get
    **silently dropped** in every subsequent sample via the ``all_field_names``
    filter in ``_generate_examples``.

    The standard ``webdataset`` package is **schema-free**: it streams every
    file in every TAR sample exactly as stored, regardless of how many images
    or other media files are present.  It also natively handles remote URLs
    (``s3://``, ``gs://``, ``http://``, ``pipe:…``) and pipe-based streaming,
    which HF's loader does not support.

    Memory-efficient raw-bytes streaming
    ------------------------------------
    Images are kept as **raw compressed bytes** (JPEG/PNG) throughout the entire
    pipeline — shuffle buffer, DataLoader workers, and prefetch queues.  A
    compressed JPEG is typically ~25 KB while its decoded PIL RGB equivalent is
    ~275 KB+ (10× larger).  Eagerly decoding via ``wds.decode("pil")`` would
    inflate every sample in the shuffle buffer and cause OOM when ``buffer_size``
    and ``num_workers`` are large.  Instead, the downstream
    ``_regularize_images()`` in ``mm_plugin.py`` lazily decodes bytes → PIL only
    when the sample is actually tokenised, keeping peak memory bounded.

    Column naming
    -------------
    Follows WebDataset's own ``base_plus_ext`` convention, identical to HF's:
        ``samplekey.0.jpg``  →  field ``"0.jpg"``  (raw bytes)
        ``samplekey.1.jpg``  →  field ``"1.jpg"``  (raw bytes)
        ``samplekey.json``   →  field ``"json"``   (decoded to dict by normaliser)

    Corrupted samples are skipped with a warning via ``wds.warn_and_continue``.
    """
    try:
        import webdataset as wds
    except ImportError:
        raise ImportError(
            "The 'webdataset' package is required for load_from='webdataset'. "
            "Install it with: pip install webdataset"
        )

    pipeline = wds.DataPipeline(
        wds.SimpleShardList(wds_files),
        wds.tarfile_to_samples(handler=wds.warn_and_continue),
        # NOTE: **no** wds.decode("pil") here — images stay as raw compressed
        # JPEG/PNG bytes throughout the shuffle buffer, DataLoader workers, and
        # prefetch queues.  This cuts per-sample memory by ~10× compared to
        # eagerly decompressing to PIL, preventing OOM when buffer_size and
        # num_workers are large.  The downstream _regularize_images() in
        # mm_plugin lazily decodes bytes→PIL only when the sample is actually
        # tokenised.  JSON columns are decoded to dict inside
        # _normalize_webdataset_sample().
    )
    # Normalize inside the generator — NOT via IterableDataset.map() which
    # merges return dicts with the original and leaks 0.jpg / 1.jpg / __key__
    # etc. as extra columns that later crash the collator.
    for sample in pipeline:
        yield _normalize_webdataset_sample(sample)


def _iter_webdataset_single_tar(wds_file: str) -> Iterator[dict]:
    """Iterate a single WebDataset TAR shard.

    This is the per-shard entry point used with ``IterableDataset.from_generator``
    when ``gen_kwargs={"wds_file": [shard0, shard1, ...]}``.  HF datasets
    treats list values in *gen_kwargs* as shards, creating one logical shard
    per element.  Accelerate then calls ``dataset.shard(num_shards=N,
    index=rank)`` to assign different TAR files to different ranks so each
    GPU only reads its own subset — no redundant I/O.
    """
    yield from _iter_webdataset_tars([wds_file] if isinstance(wds_file, str) else list(wds_file))


def _count_wds_samples(wds_files: list[str], probe_shards: int = 5) -> int:
    """Estimate total WebDataset sample count without loading any media.

    Probes up to ``probe_shards`` representative shards (spread evenly across
    the list) by reading only their TAR member *headers* (512 B each, no image
    data extracted).  Each WDS sample has exactly one ``.json`` file, so the
    count equals the number of ``.json`` members.  The per-shard average is
    then scaled to the full shard list.

    This is intentionally approximate (assumes uniform shard sizes), which is
    sufficient for ``max_steps`` computation.  Remote URLs (``s3://``, etc.)
    are skipped; only local paths are probed.
    """
    import tarfile

    _remote_prefixes = ("s3://", "gs://", "http://", "https://", "pipe:", "hf://")
    local_files = [f for f in wds_files if not any(f.startswith(p) for p in _remote_prefixes)]
    if not local_files:
        return 0

    # Pick evenly-spaced probe shards to handle datasets where shard sizes vary
    # (e.g. the last shard is smaller).
    n = len(local_files)
    indices = sorted(set(round(i * (n - 1) / max(probe_shards - 1, 1)) for i in range(probe_shards)))
    probe_files = [local_files[i] for i in indices]

    counts: list[int] = []
    for path in probe_files:
        try:
            # 'r' mode allows seeking on uncompressed .tar (reads only headers,
            # seeks past file data — very fast on local disk).  Falls back to
            # sequential decompression for .tar.gz / .tar.bz2.
            with tarfile.open(path, "r") as tf:
                counts.append(sum(1 for m in tf.getmembers() if m.name.lower().endswith(".json")))
        except Exception as exc:
            logger.warning_rank0(f"Could not probe shard {path} for sample count: {exc}")

    if not counts:
        return 0

    avg_per_shard = sum(counts) / len(counts)
    return round(avg_per_shard * n)


def _normalize_webdataset_sample(example: dict) -> dict:
    """Flatten JSON metadata and group per-sample media into extension-keyed lists.

    WebDataset TAR shards produce one column per file in the sample.  The HF
    datasets WebDataset loader uses the **full filename** as the column key
    (e.g. ``"0.jpg"``, ``"1.jpg"``, ``"0.mp4"``), not just the bare extension.
    This function:

    1. Groups all files whose extension is in ``_WDS_MEDIA_EXTENSIONS`` into
       lists keyed by extension only (e.g. ``"jpg": [bytes0, bytes1]``).  This
       lets the ``columns`` mapping in *dataset_info.json* use
       ``"images": "jpg"`` regardless of how many images each sample contains.
    2. Promotes keys from the ``"json"`` column (parsed dict) into top-level
       columns so the downstream ``columns`` mapping works naturally.
    3. Drops WebDataset-internal columns (``__key__``, ``__url__``).
    """
    result: dict = {}
    media_by_ext: dict[str, list] = {}

    for key, value in example.items():
        if key in _WDS_SKIP_COLUMNS:
            continue
        if key == "json":
            # Promote all keys from the JSON metadata into the top level.
            if isinstance(value, (bytes, bytearray)):
                value = _json_module.loads(value)
            if isinstance(value, dict):
                for k, v in value.items():
                    result[k] = v
            # String or None edge-cases are silently dropped.
        else:
            # Derive the extension: "0.jpg" → "jpg", "mp4" → "mp4"
            ext = key.rsplit(".", 1)[-1].lower() if "." in key else key
            if ext in _WDS_MEDIA_EXTENSIONS:
                # Accumulate into a list so multiple images/videos are ordered.
                if value is not None:
                    media_by_ext.setdefault(ext, []).append(value)
            else:
                result[key] = value

    result.update(media_by_ext)
    return result


def _load_single_dataset(
    dataset_attr: "DatasetAttr",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
) -> Union["Dataset", "IterableDataset"]:
    r"""Load a single dataset and aligns it to the standard format."""
    logger.info_rank0(f"Loading dataset {dataset_attr}...")
    data_path, data_name, data_dir, data_files = None, None, None, None
    if dataset_attr.load_from in ["hf_hub", "ms_hub", "om_hub"]:
        data_path = dataset_attr.dataset_name
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "script":
        data_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        data_name = dataset_attr.subset
        data_dir = dataset_attr.folder

    elif dataset_attr.load_from == "cloud_file":
        data_path = dataset_attr.dataset_name

    elif dataset_attr.load_from == "file":
        data_files = []
        local_path = os.path.join(data_args.dataset_dir, dataset_attr.dataset_name)
        if os.path.isdir(local_path):  # is directory
            for file_name in os.listdir(local_path):
                data_files.append(os.path.join(local_path, file_name))
        elif os.path.isfile(local_path):  # is file
            data_files.append(local_path)
        else:
            raise ValueError(f"File {local_path} not found.")

        data_path = FILEEXT2TYPE.get(os.path.splitext(data_files[0])[-1][1:], None)
        if data_path is None:
            raise ValueError("Allowed file types: {}.".format(",".join(FILEEXT2TYPE.keys())))

        if any(data_path != FILEEXT2TYPE.get(os.path.splitext(data_file)[-1][1:], None) for data_file in data_files):
            raise ValueError("File types should be identical.")
    elif dataset_attr.load_from == "webdataset":
        pass  # shard resolution and loading are handled in the block below
    elif dataset_attr.load_from == "lerobot":
        pass  # JSONL resolution and loading are handled in the block below
    else:
        raise NotImplementedError(f"Unknown load type: {dataset_attr.load_from}.")

    if dataset_attr.load_from == "ms_hub":
        check_version("modelscope>=1.14.0", mandatory=True)
        from modelscope import MsDataset  # type: ignore
        from modelscope.utils.config_ds import MS_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or MS_DATASETS_CACHE
        dataset = MsDataset.load(
            dataset_name=data_path,
            subset_name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.ms_hub_token,
            use_streaming=data_args.streaming,
        )
        if isinstance(dataset, MsDataset):
            dataset = dataset.to_hf_dataset()

    elif dataset_attr.load_from == "om_hub":
        check_version("openmind>=0.8.0", mandatory=True)
        from openmind import OmDataset  # type: ignore
        from openmind.utils.hub import OM_DATASETS_CACHE  # type: ignore

        cache_dir = model_args.cache_dir or OM_DATASETS_CACHE
        dataset = OmDataset.load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=cache_dir,
            token=model_args.om_hub_token,
            streaming=data_args.streaming,
        )
    elif dataset_attr.load_from == "cloud_file":
        dataset = Dataset.from_list(read_cloud_json(data_path), split=dataset_attr.split)
    elif dataset_attr.load_from == "webdataset":
        # Resolve shard pattern: join with dataset_dir when it is a bare relative path.
        shard_pattern = dataset_attr.dataset_name
        _remote_prefixes = ("s3://", "gs://", "http://", "https://", "pipe:", "hf://")
        if not os.path.isabs(shard_pattern) and not any(shard_pattern.startswith(p) for p in _remote_prefixes):
            shard_pattern = os.path.join(data_args.dataset_dir, shard_pattern)

        # Expand shell-style glob patterns (e.g. /data/shards-*.tar).
        if any(c in shard_pattern for c in "*?[{"):
            wds_files = sorted(_glob_module.glob(shard_pattern))
        else:
            wds_files = [shard_pattern]

        if not wds_files:
            raise ValueError(f"No WebDataset shard files found matching: {shard_pattern}")

        # Shuffle shard order so different runs/workers see shards in different orders.
        # This is done at load time (before the streaming pipeline is built) using the
        # global training seed.  Per-step sample-level shuffling is applied below via
        # a buffer shuffle, which is independent of -- and complements -- shard shuffling.
        rng = np.random.default_rng(training_args.seed)
        wds_files = [wds_files[i] for i in rng.permutation(len(wds_files))]

        logger.info_rank0(f"Found {len(wds_files)} WebDataset shard(s) for dataset {dataset_attr}.")
        if not data_args.streaming:
            logger.warning_rank0(
                f"Dataset {dataset_attr} is a WebDataset and will always be loaded as an IterableDataset. "
                "Consider setting `streaming: true` in your training config."
            )

        dataset = IterableDataset.from_generator(
            _iter_webdataset_single_tar,
            # Passing wds_files as a list value in gen_kwargs tells HF datasets
            # to create one logical shard per TAR file (n_shards=len(wds_files)).
            # Accelerate then uses dataset.shard(num_shards=num_processes,
            # index=rank) when n_shards > num_processes, assigning different
            # TARs to different ranks — each GPU reads only its own shards.
            # With a single shard (the default), accelerate falls back to
            # IterableDatasetShard which skips elements at the Python level
            # after reading them, wasting I/O on all ranks.
            gen_kwargs={"wds_file": wds_files},
        )
        # Normalization (JSON flattening + media grouping) is done inside
        # _iter_webdataset_tars itself.  Using .map() here would only MERGE
        # the returned dict on top of the original, leaking raw WDS columns
        # (0.jpg, 1.jpg, __key__, __url__, json) that later crash the
        # collator or get inconsistent remove_columns treatment.
        #
        # Apply a sample-level buffer shuffle unconditionally.  When data_args.streaming
        # is True, split_dataset() will call .shuffle() again (which is idempotent but
        # refreshes the buffer seed); when streaming is False, split_dataset() skips the
        # shuffle because it checks data_args.streaming -- so we must do it here.
        dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=training_args.seed)
    elif dataset_attr.load_from == "lerobot":
        from .lerobot_loader import load_lerobot_jsonl_as_dataset, make_lerobot_transform

        # Resolve JSONL file pattern: join with dataset_dir when relative.
        jsonl_pattern = dataset_attr.dataset_name
        if not os.path.isabs(jsonl_pattern):
            jsonl_pattern = os.path.join(data_args.dataset_dir, jsonl_pattern)

        # Expand glob patterns (e.g. /data/lerobot-*.jsonl).
        if any(c in jsonl_pattern for c in "*?[{"):
            jsonl_files = sorted(_glob_module.glob(jsonl_pattern))
        else:
            jsonl_files = [jsonl_pattern]

        if not jsonl_files:
            raise ValueError(f"No LeRobot JSONL files found matching: {jsonl_pattern}")

        default_dataset = dataset_attr.lerobot_default_dataset or ""
        default_camera = dataset_attr.lerobot_default_camera or "observation.images.front"

        # 1. Load JSONL as a cheap map-style Dataset (just text, no media).
        dataset = load_lerobot_jsonl_as_dataset(jsonl_files)
        logger.info_rank0(
            f"LeRobot dataset {dataset_attr}: loaded {len(dataset)} samples "
            f"from {len(jsonl_files)} JSONL file(s) as map-style Dataset."
        )

        # 2. Shuffle globally on the map-style dataset (true random access).
        dataset = dataset.shuffle(seed=training_args.seed)

        # 3. Convert to IterableDataset with enough shards for multi-GPU/worker.
        num_shards = max(
            training_args.dataloader_num_workers * max(training_args.world_size, 1),
            max(training_args.world_size, 1),
            1,
        )
        # Identify lerobot_* columns to drop after the transform.
        lerobot_cols = [c for c in dataset.column_names if c.startswith("lerobot_")]
        dataset = dataset.to_iterable_dataset(num_shards=num_shards)

        # 4. Apply lazy per-sample transform that resolves LeRobot refs → bytes.
        #    The heavy I/O (video decode, ffmpeg) runs in DataLoader workers.
        transform_fn = make_lerobot_transform(default_dataset, default_camera)
        dataset = dataset.map(transform_fn, remove_columns=lerobot_cols)
    else:
        dataset = load_dataset(
            path=data_path,
            name=data_name,
            data_dir=data_dir,
            data_files=data_files,
            split=dataset_attr.split,
            cache_dir=model_args.cache_dir,
            token=model_args.hf_hub_token,
            num_proc=data_args.preprocessing_num_workers,
            streaming=data_args.streaming and dataset_attr.load_from != "file",
        )
        if data_args.streaming and dataset_attr.load_from == "file":
            dataset = dataset.to_iterable_dataset(num_shards=training_args.dataloader_num_workers)

    if dataset_attr.num_samples is not None:
        if isinstance(dataset, IterableDataset):
            dataset = dataset.take(dataset_attr.num_samples)
            logger.info_rank0(
                f"Took first {dataset_attr.num_samples} examples from IterableDataset {dataset_attr} "
                f"(shuffle beforehand if random subsetting is needed)."
            )
        else:
            target_num = dataset_attr.num_samples
            indexes = np.random.permutation(len(dataset))[:target_num]  # all samples should be included
            target_num -= len(indexes)
            if target_num > 0:
                expand_indexes = np.random.choice(len(dataset), target_num)
                indexes = np.concatenate((indexes, expand_indexes), axis=0)

            assert len(indexes) == dataset_attr.num_samples, "Sample num mismatched."
            dataset = dataset.select(indexes)
            logger.info_rank0(f"Sampled {dataset_attr.num_samples} examples from dataset {dataset_attr}.")

    if data_args.max_samples is not None and not isinstance(dataset, IterableDataset):  # truncate dataset
        max_samples = min(data_args.max_samples, len(dataset))
        dataset = dataset.select(range(max_samples))

    return align_dataset(dataset, dataset_attr, data_args, training_args)


def _get_merged_dataset(
    dataset_names: list[str] | None,
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    return_dict: bool = False,
) -> Union["Dataset", "IterableDataset", dict[str, "Dataset"]] | None:
    r"""Return the merged datasets in the standard format."""
    if dataset_names is None:
        return None

    datasets = {}
    for dataset_name, dataset_attr in zip(dataset_names, get_dataset_list(dataset_names, data_args.dataset_dir)):
        if (stage == "rm" and dataset_attr.ranking is False) or (stage != "rm" and dataset_attr.ranking is True):
            raise ValueError("The dataset is not applicable in the current training stage.")

        datasets[dataset_name] = _load_single_dataset(dataset_attr, model_args, data_args, training_args)

    if return_dict:
        return datasets
    else:
        return merge_dataset(list(datasets.values()), data_args, seed=training_args.seed)


def _get_dataset_processor(
    data_args: "DataArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"],
    do_generate: bool = False,
) -> "DatasetProcessor":
    r"""Return the corresponding dataset processor."""
    if stage == "pt":
        dataset_processor_class = PretrainDatasetProcessor
    elif stage == "sft" and not do_generate:
        if data_args.packing:
            if data_args.neat_packing:  # hack datasets to have int32 attention mask
                from datasets.arrow_writer import OptimizedTypedSequence, TypedSequence

                def __init__(self, data, **kwargs):
                    return TypedSequence.__init__(
                        self,
                        data,
                        type=kwargs.pop("type", None),
                        try_type=kwargs.pop("try_type", None),
                        optimized_int_type=kwargs.pop("optimized_int_type", None),
                    )

                OptimizedTypedSequence.__init__ = __init__
            dataset_processor_class = PackedSupervisedDatasetProcessor
        else:
            dataset_processor_class = SupervisedDatasetProcessor

    elif stage == "rm":
        dataset_processor_class = PairwiseDatasetProcessor
    elif stage == "kto":
        dataset_processor_class = FeedbackDatasetProcessor
    else:
        dataset_processor_class = UnsupervisedDatasetProcessor

    return dataset_processor_class(template=template, tokenizer=tokenizer, processor=processor, data_args=data_args)


def _get_preprocessed_dataset(
    dataset: Union["Dataset", "IterableDataset"] | None,
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    template: "Template",
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
    is_eval: bool = False,
) -> Union["Dataset", "IterableDataset"] | None:
    r"""Preprocesses the dataset, including format checking and tokenization."""
    if dataset is None:
        return None

    dataset_processor = _get_dataset_processor(
        data_args, stage, template, tokenizer, processor, do_generate=(training_args.predict_with_generate and is_eval)
    )




    column_names = list(next(iter(dataset)).keys())
    kwargs = {}
    if not isinstance(dataset, IterableDataset):
        kwargs = dict(
            num_proc=data_args.preprocessing_num_workers,
            load_from_cache_file=(not data_args.overwrite_cache) or (training_args.local_process_index != 0),
            desc="Running tokenizer on dataset",
        )


    dataset = dataset.map(
        dataset_processor.preprocess_dataset,
        batched=True,
        batch_size=data_args.preprocessing_batch_size,
        remove_columns=column_names,
        **kwargs,
    )

    if training_args.should_log:
        try:
            print("eval example:" if is_eval else "training example:")
            dataset_processor.print_data_example(next(iter(dataset)))
        except StopIteration:
            if stage == "pt":
                raise RuntimeError("Cannot find sufficient samples, consider increasing dataset size.")
            else:
                raise RuntimeError("Cannot find valid samples, check `data/README.md` for the data format.")

    return dataset


def get_dataset(
    template: "Template",
    model_args: "ModelArguments",
    data_args: "DataArguments",
    training_args: "Seq2SeqTrainingArguments",
    stage: Literal["pt", "sft", "rm", "ppo", "kto"],
    tokenizer: "PreTrainedTokenizer",
    processor: Optional["ProcessorMixin"] = None,
) -> "DatasetModule":
    r"""Get the train dataset and optionally gets the evaluation dataset."""
    # Load tokenized dataset if path exists
    if data_args.tokenized_path is not None:
        if has_tokenized_data(data_args.tokenized_path):
            logger.warning_rank0("Loading dataset from disk will ignore other data arguments.")
            tokenized_data = load_from_disk(data_args.tokenized_path)
            dataset_module = get_dataset_module(tokenized_data)
            if data_args.streaming:
                dataset_module["train_dataset"] = dataset_module["train_dataset"].to_iterable_dataset()

            logger.info_rank0(f"Loaded tokenized dataset from {data_args.tokenized_path}.")
            return dataset_module

        if data_args.streaming:
            raise ValueError("Turn off `streaming` when saving dataset to disk.")

    # -----------------------------------------------------------------------
    # Auto-compute max_steps for WebDataset / LeRobot (IterableDataset has no __len__).
    # HF Trainer raises ValueError when train_dataset is an IterableDataset
    # and max_steps == -1.  We resolve this by probing the TAR shard headers
    # or counting JSONL lines (fast — no media data is read) and computing
    # max_steps from the sample count, num_train_epochs, and effective batch size.
    # -----------------------------------------------------------------------
    if training_args.max_steps == -1 and data_args.dataset:
        _remote_prefixes = ("s3://", "gs://", "http://", "https://", "pipe:", "hf://")
        all_attrs = get_dataset_list(data_args.dataset, data_args.dataset_dir)
        wds_attrs = [attr for attr in all_attrs if attr.load_from == "webdataset"]
        lerobot_attrs = [attr for attr in all_attrs if attr.load_from == "lerobot"]
        if wds_attrs or lerobot_attrs:
            total_samples = 0
            has_remote = False
            for attr in wds_attrs:
                shard_pattern = attr.dataset_name
                if not os.path.isabs(shard_pattern) and not any(
                    shard_pattern.startswith(p) for p in _remote_prefixes
                ):
                    shard_pattern = os.path.join(data_args.dataset_dir, shard_pattern)
                if any(shard_pattern.startswith(p) for p in _remote_prefixes):
                    has_remote = True
                    continue
                wds_files = (
                    sorted(_glob_module.glob(shard_pattern))
                    if any(c in shard_pattern for c in "*?[{")
                    else [shard_pattern]
                )
                n = _count_wds_samples(wds_files)
                if attr.num_samples is not None:
                    n = min(n, attr.num_samples)
                total_samples += n
                logger.info_rank0(
                    f"WebDataset '{attr.dataset_name}': estimated {n} samples "
                    f"across {len(wds_files)} shard(s)."
                )

            # --- LeRobot JSONL sample counting ---
            from .lerobot_loader import _count_lerobot_samples

            for attr in lerobot_attrs:
                jsonl_pattern = attr.dataset_name
                if not os.path.isabs(jsonl_pattern):
                    jsonl_pattern = os.path.join(data_args.dataset_dir, jsonl_pattern)
                jsonl_files = (
                    sorted(_glob_module.glob(jsonl_pattern))
                    if any(c in jsonl_pattern for c in "*?[{")
                    else [jsonl_pattern]
                )
                n = _count_lerobot_samples(jsonl_files)
                if attr.num_samples is not None:
                    n = min(n, attr.num_samples)
                total_samples += n
                logger.info_rank0(
                    f"LeRobot '{attr.dataset_name}': estimated {n} samples "
                    f"across {len(jsonl_files)} file(s)."
                )

            if total_samples > 0:
                if data_args.max_samples is not None:
                    total_samples = min(total_samples, data_args.max_samples)
                eff_batch = (
                    training_args.per_device_train_batch_size
                    * training_args.gradient_accumulation_steps
                    * max(training_args.world_size, 1)
                )
                max_steps = math.ceil(training_args.num_train_epochs * total_samples / eff_batch)
                training_args.max_steps = max_steps
                logger.info_rank0(
                    f"IterableDataset auto max_steps={max_steps} "
                    f"({total_samples} samples × {training_args.num_train_epochs} epoch(s) "
                    f"÷ effective batch {eff_batch}). "
                    "Override with 'max_steps' in your training config."
                )
                if has_remote:
                    logger.warning_rank0(
                        "Some WebDataset shards are remote URLs and were excluded from the "
                        "sample count. Set 'max_steps' manually for accurate scheduling."
                    )
                if data_args.packing:
                    logger.warning_rank0(
                        "'packing' is enabled: the actual number of packed sequences will "
                        "differ from the raw sample count. Consider setting 'max_steps' manually."
                    )
            elif has_remote:
                raise ValueError(
                    "All WebDataset shards are remote URLs; cannot estimate sample count. "
                    "Please set 'max_steps' explicitly in your training config."
                )

            # ── Disable dispatch_batches for IterableDataset ─────────────
            # With IterableDataset, accelerate defaults dispatch_batches=True
            # which makes the main process fetch N batches and torch.cat them
            # before dispatching to workers.  VLM batches have variable-length
            # pixel_values / input_ids, so cat fails with size mismatch.
            # Setting dispatch_batches=False lets each process fetch its own
            # batch independently (standard DDP behaviour).
            if training_args.accelerator_config.dispatch_batches is None:
                training_args.accelerator_config.dispatch_batches = False
                logger.info_rank0(
                    "IterableDataset: set dispatch_batches=False (variable-length VLM batches "
                    "are incompatible with accelerate's batch dispatching)."
                )

    # Load and preprocess dataset
    with training_args.main_process_first(desc="load dataset", local=(not data_args.data_shared_file_system)):
        dataset = _get_merged_dataset(data_args.dataset, model_args, data_args, training_args, stage)
        eval_dataset = _get_merged_dataset(
            data_args.eval_dataset,
            model_args,
            data_args,
            training_args,
            stage,
            return_dict=data_args.eval_on_each_dataset,
        )

    with training_args.main_process_first(desc="pre-process dataset", local=(not data_args.data_shared_file_system)):
        # move front to make sure eval_dataset(if contain or split) can preprocessed appropriately
        train_dict, eval_dict = split_dataset(dataset, eval_dataset, data_args, seed=training_args.seed)

        if "train" in train_dict:
            train_dict["train"] = _get_preprocessed_dataset(
                train_dict["train"], data_args, training_args, stage, template, tokenizer, processor, is_eval=False
            )

        for key in eval_dict:
            eval_dict[key] = _get_preprocessed_dataset(
                eval_dict[key], data_args, training_args, stage, template, tokenizer, processor, is_eval=True
            )

        # Combine train and eval dictionaries
        dataset_dict = DatasetDict({**train_dict, **eval_dict})

        if data_args.tokenized_path is not None:  # save tokenized dataset to disk
            if training_args.should_save:
                dataset_dict.save_to_disk(data_args.tokenized_path)
                logger.info_rank0(f"Tokenized dataset is saved at {data_args.tokenized_path}.")
                logger.info_rank0(f"Please launch the training with `tokenized_path: {data_args.tokenized_path}`.")

        return get_dataset_module(dataset_dict)
