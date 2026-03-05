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

"""Utilities for resolving ``lance://`` binary-blob URIs.

URI format: ``lance://<path>#<column>#<row>``

The handle cache is per-process (empty after fork), so each DataLoader
worker opens its own lance dataset handle and then reuses it for all
subsequent lookups — O(1) per sample after the first access.
"""

from __future__ import annotations

import threading
from typing import Any


_LANCE_HANDLES: dict[str, Any] = {}
_cache_lock = threading.Lock()


def _parse_lance_uri(uri: str) -> tuple[str, str, int]:
    """Parse ``lance://<path>#<col>#<row>`` → ``(path, col, row)``."""
    body = uri[len("lance://"):]
    path, col, row = body.rsplit("#", 2)
    return path, col, int(row)


def _get_lance_dataset(path: str) -> Any:
    """Return cached lance dataset handle for this process, opening fresh if needed."""
    with _cache_lock:
        if path not in _LANCE_HANDLES:
            import lance
            _LANCE_HANDLES[path] = lance.dataset(path)
        return _LANCE_HANDLES[path]


def clear_lance_handles() -> None:
    """Close and remove all cached Lance dataset handles.

    Call this in the *main process* right before ``dataset.map()`` forks worker
    processes.  Lance dataset objects wrap Rust/C++ file descriptors that are not
    fork-safe: if a handle is open at fork time the child and parent share the
    same underlying file descriptor, which can cause silent data corruption or
    a hard crash (SIGSEGV/SIGABRT) in the worker subprocess.

    Each forked worker will re-open its own fresh handle on first access.
    """
    with _cache_lock:
        _LANCE_HANDLES.clear()


def _is_blob_column(ds, col: str) -> bool:
    """Return True if *col* is a Lance blob extension column (lance.blob.v2) or large binary column, False otherwise."""
    try:
        import pyarrow as pa
        field = ds.schema.field(col)
        #DataType(large_binary)
        is_blob = "blob" in field.name
        if is_blob:
            return True
        return isinstance(field.type, pa.ExtensionType) and "blob" in getattr(field.type, "extension_name", "").lower()
    except Exception:
        return False


def resolve_lance_uri(uri: str) -> bytes:
    """Fetch binary blob for a ``lance://`` URI.

    Dispatches to ``take_blobs()`` for Lance blob extension columns
    (``lance.blob.v2``) and falls back to ``take()`` for plain binary columns.
    O(1) after the first open per (path, process).
    """
    path, col, row = _parse_lance_uri(uri)
    ds = _get_lance_dataset(path)
    if _is_blob_column(ds, col):
        blobs = ds.take_blobs(col, indices=[row])
        data = blobs[0].read()
        blobs[0].close()
        return data
    return ds.take([row], columns=[col]).to_pylist()[0][col]
