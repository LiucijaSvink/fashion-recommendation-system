"""Disk-backed memoisation for the expensive steps of a long run.

A full six-fold run is hours of Spark work made of a few dozen independently
expensive pieces — one sweep on one fold, one trained model, one scored fold. If the
process dies at hour five, nothing about the first four hours has changed, so this
module writes each finished piece to `{out_dir}/_run_cache` and reuses it on the next
attempt.

Keys carry a fingerprint of the configuration that produced them, so changing a
setting produces a different key rather than silently reusing a stale result. Writes
go to a temporary file and are renamed into place, so a crash mid-write cannot leave
a half-written entry that a later run would trust.

    value = cached(cfg, "per_source_report-fold_0", lambda: expensive(...))
"""
from __future__ import annotations

import hashlib
import json
import logging
import pickle
from pathlib import Path
from typing import Any, Callable

from .config import PipelineConfig

log = logging.getLogger(__name__)

CACHE_DIRNAME = "_run_cache"

# Bump when the *shape* of a cached result changes — the key fingerprints the config,
# not the code, so without this a reused entry can silently hold the old format.
# (1 -> 2: rank_sweep switched from hit counts to recall %.)
CACHE_VERSION = 2


def fingerprint(cfg: PipelineConfig, extra: Any = None) -> str:
    """Short stable digest of a configuration (plus anything else that matters)."""
    payload = json.dumps({"version": CACHE_VERSION, "cfg": cfg.to_dict(), "extra": extra},
                         sort_keys=True, default=str)
    return hashlib.sha1(payload.encode()).hexdigest()[:10]


def cache_dir(cfg: PipelineConfig) -> Path:
    """Where cached results for this configuration live, created if absent."""
    directory = Path(cfg.out_dir) / CACHE_DIRNAME
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def cached(
    cfg: PipelineConfig,
    key: str,
    compute: Callable[[], Any],
    *,
    extra: Any = None,
    refresh: bool = False,
) -> Any:
    """Return `compute()`, reading from or writing to the run cache.

    `refresh=True` recomputes and overwrites, which is how to invalidate one entry
    without discarding the rest of a run.
    """
    target = cache_dir(cfg) / f"{key}.{fingerprint(cfg, extra)}.pkl"
    if target.exists() and not refresh:
        log.info("cache hit: %s", target.name)
        with target.open("rb") as handle:
            return pickle.load(handle)

    value = compute()
    staging = target.with_suffix(".writing")
    with staging.open("wb") as handle:
        pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
    staging.replace(target)          # atomic — a partial file is never left behind
    log.info("cached: %s", target.name)
    return value


def clear(cfg: PipelineConfig, pattern: str = "*") -> int:
    """Delete cache entries matching `pattern`. Returns how many were removed."""
    entries = list(cache_dir(cfg).glob(f"{pattern}.pkl"))
    for entry in entries:
        entry.unlink()
    return len(entries)
