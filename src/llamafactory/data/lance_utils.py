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

The handle cache is per-process (empty after fork).  Each DataLoader worker
opens its own fresh lance handle on first access (lance C++ state is not
fork-safe to share).  Opens are staggered by worker_id to avoid the
thundering-herd problem when many workers hit the same file simultaneously.
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
    """Return cached lance dataset handle for this process, opening fresh if needed.

    Stagger the first open by DataLoader worker_id to prevent thundering-herd:
    with fork-based workers each process starts with an empty cache and all N
    workers would otherwise call lance.dataset() simultaneously, contending on
    the lance file lock / NFS metadata server.  The sleep is done before
    acquiring _cache_lock so multiple distinct lance paths don't serialize.
    """
    if path in _LANCE_HANDLES:
        return _LANCE_HANDLES[path]
    try:
        import time
        import torch.utils.data
        info = torch.utils.data.get_worker_info()
        if info is not None:
            time.sleep(info.id * 0.3)  # stagger: worker N waits N*300ms
    except Exception:
        pass
    with _cache_lock:
        if path not in _LANCE_HANDLES:
            import lance
            _LANCE_HANDLES[path] = lance.dataset(path)
        return _LANCE_HANDLES[path]


def _is_blob_column(ds, col: str) -> bool:
    """Return True if *col* is a Lance blob extension column (lance.blob.v2)."""
    try:
        import pyarrow as pa
        field = ds.schema.field(col)
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
