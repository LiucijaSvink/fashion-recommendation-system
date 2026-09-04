"""Single, serialisable description of a pipeline run.

Every path, window, cap, ratio and seed the pipeline uses lives here, so a run is
fully described by one object (and one JSON file). Nothing downstream reads a
hard-coded path or magic number.

    from recolib.config import PipelineConfig
    cfg = PipelineConfig.from_json("configs/config.json")
    cfg.fold_dir("fold_0")        # -> /path/to/out/fold_0
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

from .schema import DEFAULT_SCHEMA, Schema


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration for one end-to-end run. Immutable; use `.with_(...)` to vary it."""

    # ---- data locations -------------------------------------------------
    raw_dir: str
    out_dir: str
    fold_prefix: str = "fold"
    submission_path: str = "submission.csv"
    # optional per-artifact directory names, e.g. {"features": "features_v4"} to read a
    # build produced by an earlier run without copying it
    artifacts: dict[str, str] = field(default_factory=dict)

    # ---- time-series splitting ------------------------------------------
    window_end: str = "2020-09-22"
    train_weeks: int = 6
    val_weeks: int = 1
    n_folds: int = 6

    # ---- retrieval -------------------------------------------------------
    max_candidates: int | None = None   # None -> keep the whole union (see section 3.3)
    seed_weeks: int = 4
    repurchase_top_n: int = 80
    itemcf_top_n: int = 100
    itemcf_neighbors: int = 50
    product_code_top_n: int = 30
    popularity_top_n: int = 100
    popularity_weeks: int = 4
    als_top_n: int = 80
    als_rank: int = 64
    als_iters: int = 10

    # ---- features --------------------------------------------------------
    feature_chunks: int = 4      # passes over the candidate table; bounds shuffle size
    user_recent_weeks: int = 12
    item_windows: tuple[int, ...] = (1, 4, 12)
    affinity_keys: tuple[tuple[str, str], ...] = (
        ("product_code", "pcode"),
        ("product_type_no", "ptype"),
        ("department_no", "dept"),
        ("product_group_name", "pgroup"),
    )

    # ---- reranker --------------------------------------------------------
    train_folds: tuple[str, ...] = ("fold_0",)
    negatives_per_positive: int = 15
    lgbm_params: dict[str, Any] = field(default_factory=dict)

    # ---- evaluation ------------------------------------------------------
    eval_k: int = 12
    eval_sample_users: int | None = None   # None -> score every customer

    # ---- runtime ---------------------------------------------------------
    seed: int = 42
    driver_memory: str = "12g"
    shuffle_partitions: int = 400
    schema: Schema = DEFAULT_SCHEMA

    # ---- derived paths ---------------------------------------------------
    @property
    def transactions_csv(self) -> str:
        return f"{self.raw_dir}/transactions_train.csv"

    @property
    def customers_csv(self) -> str:
        return f"{self.raw_dir}/customers.csv"

    @property
    def articles_csv(self) -> str:
        return f"{self.raw_dir}/articles.csv"

    @property
    def sample_submission_csv(self) -> str:
        return f"{self.raw_dir}/sample_submission.csv"

    @property
    def history_path(self) -> str:
        """Full transaction history parquet — the source of lifetime signals."""
        return f"{self.out_dir}/transactions"

    @property
    def articles_path(self) -> str:
        return f"{self.out_dir}/articles"

    @property
    def customers_path(self) -> str:
        return f"{self.out_dir}/customers"

    @property
    def fold_names(self) -> list[str]:
        return [f"{self.fold_prefix}_{i}" for i in range(self.n_folds)]

    @property
    def submission_fold(self) -> str:
        return f"{self.fold_prefix}_submission"

    @property
    def holdout_fold(self) -> str | None:
        """The fold to evaluate on: the one immediately newer than the newest
        training fold, so no label the model saw is scored.

        Lower index = more recent, so the newest training fold is `min(indices)` and
        the held-out fold sits one below it. `None` when training already includes
        fold 0 — there is nothing newer to hold out, which is correct for the model
        that ships but means the run cannot report an honest score.
        """
        indices = [int(name.rsplit("_", 1)[-1]) for name in self.train_folds
                   if name.rsplit("_", 1)[-1].isdigit()]
        if not indices or min(indices) == 0:
            return None
        return f"{self.fold_prefix}_{min(indices) - 1}"

    def fold_dir(self, fold: str) -> str:
        return f"{self.out_dir}/{fold}"

    def path(self, fold: str, artifact: str) -> str:
        """Path of a fold artifact: train | val | candidates | features."""
        return f"{self.fold_dir(fold)}/{self.artifacts.get(artifact, artifact)}"

    # ---- (de)serialisation -----------------------------------------------
    def with_(self, **changes: Any) -> "PipelineConfig":
        """Return a copy with `changes` applied (config stays immutable)."""
        unknown = set(changes) - set(self.__dataclass_fields__)
        if unknown:
            raise ValueError(
                f"unknown config field(s): {sorted(unknown)}. "
                f"valid fields: {sorted(self.__dataclass_fields__)}"
            )
        return replace(self, **changes)

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["schema"] = asdict(self.schema)
        return out

    def to_json(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2, default=list))

    @classmethod
    def from_json(cls, path: str | Path) -> "PipelineConfig":
        raw = json.loads(Path(path).read_text())
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> "PipelineConfig":
        data = dict(raw)
        if isinstance(data.get("schema"), dict):
            data["schema"] = Schema(**data["schema"])
        for key in ("raw_dir", "out_dir"):
            if isinstance(data.get(key), str):
                data[key] = os.path.expanduser(os.path.expandvars(data[key]))
        for key in ("item_windows", "train_folds"):
            if key in data and data[key] is not None:
                data[key] = tuple(data[key])
        if "affinity_keys" in data and data["affinity_keys"] is not None:
            data["affinity_keys"] = tuple(tuple(pair) for pair in data["affinity_keys"])
        unknown = set(data) - {f for f in cls.__dataclass_fields__}
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        return cls(**data)

    def cli_command(self, config_path: str = "configs/config.json",
                    stages: list[str] | None = None) -> str:
        """The bash command that reproduces this config — keeps notebook and CLI in step."""
        command = f"recolib run --config {config_path}"
        if stages:
            command += " --stages " + " ".join(stages)
        return command

    def summary(self) -> str:
        """One-screen description of the run, for the top of a notebook or a log."""
        lines = [
            f"data          {self.raw_dir} -> {self.out_dir}",
            f"folds         {self.n_folds} x ({self.train_weeks}w train + {self.val_weeks}w val), "
            f"ending {self.window_end}",
            f"retrieval     "
            + (f"cap {self.max_candidates}/customer" if self.max_candidates else "uncapped")
            + f", seeds {self.seed_weeks}w",
            f"features      user window {self.user_recent_weeks}w, item windows "
            f"{'/'.join(f'{w}w' for w in self.item_windows)}",
            f"reranker      folds {list(self.train_folds)}, "
            f"{self.negatives_per_positive}:1 negatives, seed {self.seed}",
            f"evaluation    MAP@{self.eval_k} on "
            + (f"{self.eval_sample_users:,} sampled customers" if self.eval_sample_users
               else "all customers"),
        ]
        return "\n".join(lines)
