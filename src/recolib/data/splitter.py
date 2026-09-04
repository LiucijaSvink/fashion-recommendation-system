"""Train/val/test splitting strategies.

`TimeSeriesSplitter` — rolling walk-forward folds, leakage-safe (the correct default
for next-period prediction). `RandomSplitter` — random fractions (opt-in; warns that
it leaks time for forecasting, but is handy for sanity checks / by-user cold-start eval).
Both return `list[Fold]` so downstream code treats them uniformly. Column names come
from `Schema` (defaults to H&M).
"""
from __future__ import annotations

import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from ..schema import DEFAULT_SCHEMA, Schema


@dataclass
class Fold:
    """One train/validation split of the transaction history.

    `train` is the window the fold learns from, `val` the week it is judged on, and
    `meta` carries the boundary dates. The submission fold has no `val` — there is no
    future to look at — which `is_submission` detects.
    """
    name: str
    train: DataFrame
    val: DataFrame | None = None
    test: DataFrame | None = None
    meta: dict = field(default_factory=dict)

    @property
    def is_submission(self) -> bool:
        """True for a final fold with no labelled validation (predict the future)."""
        return self.val is None or not self.val.head(1)


class Splitter(ABC):
    """Turns a transaction history into folds. Subclasses decide on what basis."""

    @abstractmethod
    def split(self, transactions: DataFrame) -> list[Fold]:
        """Return the folds, in the order the pipeline should use them."""
        raise NotImplementedError


class TimeSeriesSplitter(Splitter):
    """Walk-forward: fold i has `train_weeks` of train ending just before a 1×`val_weeks`
    validation window; fold 0's validation ends at `window_end`. Optionally appends a
    submission fold (train through `window_end`, empty val) for the real future week.

    `train_weeks=None` makes train unbounded — everything up to `train_end`. Use it when
    the question is whether a signal exists at all (the EDA notebook) rather than what a
    bounded training window can exploit (the model pipeline). Validation windows are
    identical either way, so the two uses stay comparable.
    """

    def __init__(
        self,
        window_end: str | date,
        train_weeks: int | None = 6,
        val_weeks: int = 1,
        n_folds: int = 6,
        with_submission: bool = True,
        schema: Schema = DEFAULT_SCHEMA,
    ):
        self.window_end = date.fromisoformat(window_end) if isinstance(window_end, str) else window_end
        self.train_weeks, self.val_weeks = train_weeks, val_weeks
        self.n_folds, self.with_submission = n_folds, with_submission
        self.schema = schema

    def _train_window(self, transactions: DataFrame, train_end: date):
        """Rows up to `train_end`, floored at `train_weeks` back when that is set."""
        d = self.schema.date
        if self.train_weeks is None:
            return transactions.filter(F.col(d) <= F.lit(train_end)), None
        train_start = train_end - timedelta(weeks=self.train_weeks) + timedelta(days=1)
        window = transactions.filter((F.col(d) >= F.lit(train_start)) & (F.col(d) <= F.lit(train_end)))
        return window, train_start

    def split(self, transactions: DataFrame) -> list[Fold]:
        d = self.schema.date
        folds: list[Fold] = []
        for i in range(self.n_folds):
            val_end = self.window_end - timedelta(weeks=i * self.val_weeks)
            val_start = val_end - timedelta(weeks=self.val_weeks) + timedelta(days=1)
            train_end = val_start - timedelta(days=1)
            assert train_end < val_start, "leakage: train overlaps val"
            train, train_start = self._train_window(transactions, train_end)
            folds.append(Fold(
                name=f"fold_{i}",
                train=train,
                val=transactions.filter((F.col(d) >= F.lit(val_start)) & (F.col(d) <= F.lit(val_end))),
                meta=dict(train_start=train_start, train_end=train_end, val_start=val_start, val_end=val_end),
            ))
        if self.with_submission:
            train, train_start = self._train_window(transactions, self.window_end)
            folds.append(Fold(
                name="submission",
                train=train,
                val=None,
                meta=dict(train_start=train_start, train_end=self.window_end, val_start=None, val_end=None),
            ))
        return folds


class RandomSplitter(Splitter):
    """Random train/val/test. NOT time-safe for forecasting — use for sanity checks, or
    `by="user"` for cold-start (new-user) evaluation. Returns a single Fold."""

    def __init__(
        self,
        val_frac: float = 0.1,
        test_frac: float = 0.0,
        by: str = "row",
        seed: int = 42,
        schema: Schema = DEFAULT_SCHEMA,
    ):
        assert by in ("row", "user")
        assert val_frac + test_frac < 1.0
        self.val_frac, self.test_frac, self.by, self.seed = val_frac, test_frac, by, seed
        self.schema = schema

    def split(self, transactions: DataFrame) -> list[Fold]:
        warnings.warn(
            "RandomSplitter leaks time for next-period prediction (a user's future can land "
            "in train). Use TimeSeriesSplitter for forecasting; this is for sanity/cold-start.",
            stacklevel=2,
        )
        train_frac = 1.0 - self.val_frac - self.test_frac
        if self.by == "row":
            train, val, test = transactions.randomSplit([train_frac, self.val_frac, self.test_frac], seed=self.seed)
        else:  # by user
            u = self.schema.user
            users = transactions.select(u).distinct()
            u_tr, u_va, u_te = users.randomSplit([train_frac, self.val_frac, self.test_frac], seed=self.seed)
            train = transactions.join(F.broadcast(u_tr), on=u, how="inner")
            val = transactions.join(F.broadcast(u_va), on=u, how="inner")
            test = transactions.join(F.broadcast(u_te), on=u, how="inner")
        folds = [Fold(name="random", train=train, val=val,
                      test=(test if self.test_frac > 0 else None),
                      meta=dict(by=self.by, val_frac=self.val_frac, test_frac=self.test_frac))]
        return folds
