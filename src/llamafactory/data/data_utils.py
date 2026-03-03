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

import copy
import json
from enum import StrEnum, unique
from typing import TYPE_CHECKING, Any, Optional, TypedDict, Union

import numpy as np
import fsspec
from datasets import Dataset, DatasetDict, DatasetInfo, IterableDataset, concatenate_datasets, interleave_datasets
from datasets.iterable_dataset import (
    CyclingMultiSourcesExamplesIterable,
    RandomlyCyclingMultiSourcesExamplesIterable,
    VerticallyConcatenatedMultiSourcesExamplesIterable,
)

from ..extras import logging


if TYPE_CHECKING:
    from ..hparams import DataArguments


logger = logging.get_logger(__name__)


SLOTS = list[Union[str, set[str], dict[str, str]]]


@unique
class Role(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    FUNCTION = "function"
    OBSERVATION = "observation"


class DatasetModule(TypedDict):
    train_dataset: Optional[Union["Dataset", "IterableDataset"]]
    eval_dataset: Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]]


def _features_compatible(all_datasets: list[Union["Dataset", "IterableDataset"]]) -> bool:
    r"""Return True if all dataset feature schemas can be aligned by HF."""
    try:
        from datasets.features.features import _check_if_features_can_be_aligned

        features_list = []
        for ds in all_datasets:
            if isinstance(ds, IterableDataset):
                resolved = ds._resolve_features()
                feats = resolved.features
            else:
                feats = ds.features
            if feats is not None:
                features_list.append(feats)

        if len(features_list) < 2:
            return True
        _check_if_features_can_be_aligned(features_list)
        return True
    except Exception:
        return False


def _interleave_iterable_featureless(
    datasets: list["IterableDataset"],
    probabilities: Optional[list[float]] = None,
    seed: Optional[int] = None,
    stopping_strategy: str = "first_exhausted",
) -> "IterableDataset":
    r"""Interleave ``IterableDataset`` objects **without** Arrow feature alignment.

    Uses the same HF ``CyclingMultiSourcesExamplesIterable`` /
    ``RandomlyCyclingMultiSourcesExamplesIterable`` classes that
    ``interleave_datasets`` uses internally, so multi-worker sharding, shard
    shuffling, etc. are fully preserved.  The only difference is that the
    ``_check_if_features_can_be_aligned`` check is skipped and the resulting
    dataset has ``features=None`` — ``mm_plugin`` handles type dispatch at
    runtime.
    """
    resolved = [ds._resolve_features() for ds in datasets]
    ex_iterables = [copy.deepcopy(ds._ex_iterable) for ds in resolved]

    if probabilities is None:
        ex_iterable = CyclingMultiSourcesExamplesIterable(
            ex_iterables, stopping_strategy=stopping_strategy
        )
    else:
        generator = np.random.default_rng(seed)
        ex_iterable = RandomlyCyclingMultiSourcesExamplesIterable(
            ex_iterables,
            generator=generator,
            probabilities=probabilities,
            stopping_strategy=stopping_strategy,
        )

    info = DatasetInfo.from_merge([ds.info for ds in resolved])
    info.features = None  # skip feature encoding/decoding
    token_per_repo_id = {
        repo_id: token
        for ds in resolved
        for repo_id, token in ds._token_per_repo_id.items()
    }
    return IterableDataset(
        ex_iterable=ex_iterable, info=info, split=None, token_per_repo_id=token_per_repo_id
    )


def _concatenate_iterable_featureless(
    datasets: list["IterableDataset"],
) -> "IterableDataset":
    r"""Concatenate ``IterableDataset`` objects **without** Arrow feature alignment.

    Same idea as :func:`_interleave_iterable_featureless` — uses HF's
    ``VerticallyConcatenatedMultiSourcesExamplesIterable`` directly.
    """
    resolved = [ds._resolve_features() for ds in datasets]
    ex_iterables = [copy.deepcopy(ds._ex_iterable) for ds in resolved]
    ex_iterable = VerticallyConcatenatedMultiSourcesExamplesIterable(ex_iterables)

    info = DatasetInfo.from_merge([ds.info for ds in resolved])
    info.features = None
    token_per_repo_id = {
        repo_id: token
        for ds in resolved
        for repo_id, token in ds._token_per_repo_id.items()
    }
    return IterableDataset(
        ex_iterable=ex_iterable, info=info, split=None, token_per_repo_id=token_per_repo_id
    )


def _promote_to_iterable(
    all_datasets: list[Union["Dataset", "IterableDataset"]],
) -> list[Union["Dataset", "IterableDataset"]]:
    r"""Promote any map-style Dataset to IterableDataset when the list is mixed."""
    has_iterable = any(isinstance(ds, IterableDataset) for ds in all_datasets)
    has_map = any(isinstance(ds, Dataset) for ds in all_datasets)
    if has_iterable and has_map:
        logger.warning_rank0_once(
            "Detected a mix of map-style Dataset and IterableDataset (e.g. WebDataset). "
            "Converting all map-style datasets to IterableDataset for compatibility."
        )
        return [ds.to_iterable_dataset() if isinstance(ds, Dataset) else ds for ds in all_datasets]
    return all_datasets


def merge_dataset(
    all_datasets: list[Union["Dataset", "IterableDataset"]], data_args: "DataArguments", seed: int
) -> Union["Dataset", "IterableDataset"]:
    r"""Merge multiple datasets to a unified dataset."""
    if len(all_datasets) == 1:
        return all_datasets[0]

    all_datasets = _promote_to_iterable(all_datasets)
    compatible = _features_compatible(all_datasets)

    if not compatible:
        logger.warning_rank0_once(
            "Datasets have incompatible Arrow feature schemas (e.g. `_videos` is "
            "List(Value('binary')) in WebDataset but List(Value('string')) in a path-based "
            "dataset). Using featureless merge — mm_plugin handles both bytes and "
            "string paths at runtime."
        )

    if data_args.mix_strategy == "concat":
        if data_args.streaming:
            logger.warning_rank0_once(
                "The samples between different datasets will not be mixed in streaming mode."
            )
        if compatible:
            return concatenate_datasets(all_datasets)
        else:
            return _concatenate_iterable_featureless(all_datasets)

    elif data_args.mix_strategy.startswith("interleave"):
        if not data_args.streaming:
            logger.warning_rank0_once("We recommend using `mix_strategy=concat` in non-streaming mode.")

        strategy_map: str = {
            "interleave_under": "first_exhausted",
            "interleave_over": "all_exhausted",
            "interleave_once": "all_exhausted_without_replacement",
        }[data_args.mix_strategy]

        if compatible:
            return interleave_datasets(
                datasets=all_datasets,
                probabilities=data_args.interleave_probs,
                seed=seed,
                stopping_strategy=strategy_map,  # type: ignore
            )
        else:
            return _interleave_iterable_featureless(
                datasets=all_datasets,
                probabilities=data_args.interleave_probs,
                seed=seed,
                stopping_strategy=strategy_map,
            )

    else:
        raise ValueError(f"Unknown mixing strategy: {data_args.mix_strategy}.")


def split_dataset(
    dataset: Optional[Union["Dataset", "IterableDataset"]],
    eval_dataset: Optional[Union["Dataset", "IterableDataset", dict[str, "Dataset"]]],
    data_args: "DataArguments",
    seed: int,
) -> tuple[dict, dict]:
    r"""Split the dataset and returns two dicts containing train set and validation set.

    Support both map dataset and iterable dataset.

    Returns:
        train_dict: Dictionary containing training data with key "train"
        eval_dict: Dictionary containing evaluation data with keys "validation" or "validation_{name}"
    """
    if eval_dataset is not None and data_args.val_size > 1e-6:
        raise ValueError("Cannot specify `val_size` if `eval_dataset` is not None.")

    # the train and eval better to in dict dtype and separately return for cpode clearly and good handle outside
    train_dict, eval_dict = {}, {}

    if dataset is not None:
        if isinstance(dataset, IterableDataset):
            dataset = dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)

        if data_args.val_size > 1e-6:
            if isinstance(dataset, IterableDataset):
                eval_dict["validation"] = dataset.take(int(data_args.val_size))
                train_dict["train"] = dataset.skip(int(data_args.val_size))
            else:
                val_size = int(data_args.val_size) if data_args.val_size > 1 else data_args.val_size
                split_result = dataset.train_test_split(test_size=val_size, seed=seed)
                train_dict["train"] = split_result["train"]
                eval_dict["validation"] = split_result["test"]
        else:
            train_dict["train"] = dataset

    if eval_dataset is not None:
        if isinstance(eval_dataset, dict):
            for name, data in eval_dataset.items():
                eval_dict[f"validation_{name}"] = data
        else:
            if isinstance(eval_dataset, IterableDataset):
                eval_dataset = eval_dataset.shuffle(buffer_size=data_args.buffer_size, seed=seed)

            eval_dict["validation"] = eval_dataset

    return train_dict, eval_dict


def get_dataset_module(dataset: Union["Dataset", "DatasetDict"]) -> "DatasetModule":
    r"""Convert dataset or dataset dict to dataset module."""
    dataset_module: DatasetModule = {}
    if isinstance(dataset, DatasetDict):  # dataset dict
        if "train" in dataset:
            dataset_module["train_dataset"] = dataset["train"]

        if "validation" in dataset:
            dataset_module["eval_dataset"] = dataset["validation"]
        else:
            eval_dataset = {}
            for key in dataset.keys():
                if key.startswith("validation_"):
                    eval_dataset[key[len("validation_") :]] = dataset[key]

            if len(eval_dataset):
                dataset_module["eval_dataset"] = eval_dataset

    else:  # single dataset
        dataset_module["train_dataset"] = dataset

    return dataset_module


def setup_fs(path: str, anon: bool = False) -> "fsspec.AbstractFileSystem":
    r"""Set up a filesystem object based on the path protocol."""
    storage_options = {"anon": anon} if anon else {}
    if path.startswith("s3://"):
        fs = fsspec.filesystem("s3", **storage_options)
    elif path.startswith(("gs://", "gcs://")):
        fs = fsspec.filesystem("gcs", **storage_options)
    else:
        raise ValueError(f"Unsupported protocol in path: {path}. Use 's3://' or 'gs://'.")

    if not fs.exists(path):
        raise ValueError(f"Path does not exist: {path}.")

    return fs


def _read_json_with_fs(fs: "fsspec.AbstractFileSystem", path: str) -> list[Any]:
    r"""Helper function to read JSON/JSONL files using fsspec."""
    with fs.open(path, "r") as f:
        if path.endswith(".jsonl"):
            return [json.loads(line) for line in f if line.strip()]
        else:
            return json.load(f)


def read_cloud_json(cloud_path: str) -> list[Any]:
    r"""Read a JSON/JSONL file from cloud storage (S3 or GCS).

    Args:
        cloud_path: str
            Cloud path in the format:
            - 's3://bucket-name/file.json' for AWS S3
            - 'gs://bucket-name/file.jsonl' or 'gcs://bucket-name/file.jsonl' for Google Cloud Storage
    """
    try:
        fs = setup_fs(cloud_path, anon=True)  # try with anonymous access first
    except Exception:
        fs = setup_fs(cloud_path)  # try again with credentials

    # filter out non-JSON files
    files = [x["Key"] for x in fs.listdir(cloud_path)] if fs.isdir(cloud_path) else [cloud_path]
    files = filter(lambda file: file.endswith(".json") or file.endswith(".jsonl"), files)
    if not files:
        raise ValueError(f"No JSON/JSONL files found in the specified path: {cloud_path}.")

    return sum([_read_json_with_fs(fs, file) for file in files], [])
