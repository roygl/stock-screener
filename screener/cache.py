"""Local, date-keyed cache for the data layer.

End-of-day data changes at most once per calendar day, so the cache is keyed by
``(namespace, key, date)``: the first fetch on a given day hits the network and
writes a file; every later read that same day is served from disk. The next day
the key misses and the provider refetches.

Two payload shapes are supported:
- DataFrames        -> Parquet  (price history)
- JSON-able ``dict`` -> JSON    (fundamentals, earnings dates)

Files live under ``.cache/<namespace>/<key>__<YYYY-MM-DD>.<ext>`` (``.cache`` is
ignored per CLAUDE.md). Writes are atomic (temp file + ``os.replace``) and prune
older-dated files for the same key, so the cache holds only the current day.
"""

from __future__ import annotations

import datetime as dt
import json
import os
import re
import tempfile
from pathlib import Path

import pandas as pd

# .cache/ lives one level up from this file's package (repo root).
CACHE_ROOT = Path(__file__).resolve().parent.parent / ".cache"

# Anything outside this set is replaced in filenames so keys like "BRK.B" stay safe.
_UNSAFE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe(key: str) -> str:
    return _UNSAFE.sub("_", key.strip())


class Cache:
    """Date-keyed file cache. One instance is cheap; share it across a run."""

    def __init__(self, root: str | Path = CACHE_ROOT) -> None:
        self.root = Path(root)

    # --- DataFrame payloads (Parquet) ------------------------------------
    def get_frame(self, namespace: str, key: str, *, on_date: dt.date | None = None) -> pd.DataFrame | None:
        path = self._path(namespace, key, "parquet", on_date)
        if not path.exists():
            return None
        try:
            return pd.read_parquet(path)
        except Exception:
            # Corrupt/partial file: treat as a miss so the caller refetches.
            return None

    def put_frame(self, namespace: str, key: str, df: pd.DataFrame, *, on_date: dt.date | None = None) -> None:
        path = self._path(namespace, key, "parquet", on_date)
        self._atomic_write(path, lambda p: df.to_parquet(p))
        self._prune(namespace, key, "parquet", keep=path.name)

    # --- JSON payloads ---------------------------------------------------
    def get_json(self, namespace: str, key: str, *, on_date: dt.date | None = None) -> dict | None:
        path = self._path(namespace, key, "json", on_date)
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except Exception:
            return None

    def put_json(self, namespace: str, key: str, obj: dict, *, on_date: dt.date | None = None) -> None:
        path = self._path(namespace, key, "json", on_date)
        self._atomic_write(path, lambda p: p.write_text(json.dumps(obj, default=str)))
        self._prune(namespace, key, "json", keep=path.name)

    # --- maintenance -----------------------------------------------------
    def clear(self, namespace: str | None = None) -> None:
        """Delete cached files (all, or one namespace). Mainly for tests."""
        target = self.root if namespace is None else self.root / namespace
        if not target.exists():
            return
        for f in target.rglob("*"):
            if f.is_file():
                f.unlink()

    # --- internals -------------------------------------------------------
    def _path(self, namespace: str, key: str, ext: str, on_date: dt.date | None) -> Path:
        day = (on_date or dt.date.today()).isoformat()
        return self.root / namespace / f"{_safe(key)}__{day}.{ext}"

    @staticmethod
    def _atomic_write(path: Path, write) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
        os.close(fd)
        tmp_path = Path(tmp)
        try:
            write(tmp_path)
            os.replace(tmp_path, path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    def _prune(self, namespace: str, key: str, ext: str, *, keep: str) -> None:
        folder = self.root / namespace
        if not folder.exists():
            return
        for f in folder.glob(f"{_safe(key)}__*.{ext}"):
            if f.name != keep:
                f.unlink()
