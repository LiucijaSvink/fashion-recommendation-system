"""Ranking metrics for recommender evaluation (pure Python / numpy — no Spark).

Why pure Python and not PySpark?
    These metrics operate on already-aggregated data — `{user -> top-k items}` —
    which is small. After the reranker produces top-k per customer that's a few
    million small lists; pure-Python list comprehensions handle it in 1–3 seconds.
    PySpark would add 30+ seconds of JVM round-trip / partition-coordination
    overhead for the same data. The break-even point where Spark wins is ~100M+
    rows of raw input, not applicable for already-ranked recommendations.

The module is organised into three families:
  * Single-user helpers (private + the public `apk`).
  * Mean-over-users public metrics, all sharing the same shape:
        mapk, ndcg_at_k, precision_at_k, recall_at_k, mrr_at_k, hit_rate_at_k.
    Each accepts an optional `users` parameter so you can either:
        - average over the customers who appear in `predictions` (default), or
        - average over a fixed population (e.g. the full Kaggle sample submission
          including non-buyers, which drags the score down — Kaggle-style MAP).
  * `hit_rate` — a *retrieval-stage* metric, with a different (set-based) shape;
    used to evaluate candidate coverage BEFORE recommendations exist.
"""
from __future__ import annotations

from typing import Callable, Iterable, Mapping, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# Single-user helpers
# ---------------------------------------------------------------------------

def apk(actual: set, predicted: Sequence, k: int = 12) -> float:
    """Average precision @ k for a single user.

    Matches the Kaggle H&M MAP@12 convention: divide by min(len(actual), k).
    """
    if not actual:
        return 0.0
    score, hits, seen = 0.0, 0, set()
    for i, p in enumerate(predicted[:k]):
        if p in actual and p not in seen:
            hits += 1
            score += hits / (i + 1.0)
        seen.add(p)
    return score / min(len(actual), k)


def _ndcg_user(actual: set, predicted: Sequence, k: int) -> float:
    """NDCG @ k for a single user (binary relevance: hit=1, miss=0)."""
    if not actual:
        return 0.0
    seen, dcg = set(), 0.0
    for i, p in enumerate(predicted[:k]):
        if p in actual and p not in seen:
            dcg += 1.0 / np.log2(i + 2)
        seen.add(p)
    ideal_n = min(len(actual), k)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(ideal_n))
    return dcg / idcg if idcg > 0 else 0.0


def _precision_user(actual: set, predicted: Sequence, k: int) -> float:
    """Precision @ k for a single user."""
    if not predicted:
        return 0.0
    seen, hits = set(), 0
    for p in predicted[:k]:
        if p in actual and p not in seen:
            hits += 1
        seen.add(p)
    return hits / min(k, len(predicted))


def _recall_user(actual: set, predicted: Sequence, k: int) -> float:
    """Recall @ k for a single user. Returns 0 if `actual` is empty."""
    if not actual:
        return 0.0
    return len(set(predicted[:k]) & actual) / len(actual)


def _mrr_user(actual: set, predicted: Sequence, k: int) -> float:
    """Reciprocal rank of the first relevant item in top-k. 0 if no hit."""
    for i, p in enumerate(predicted[:k]):
        if p in actual:
            return 1.0 / (i + 1)
    return 0.0


# ---------------------------------------------------------------------------
# Mean-over-population aggregator (used by every public metric below)
# ---------------------------------------------------------------------------

def _mean_over_users(
    predictions: Mapping,
    ground_truth: Mapping,
    k: int,
    users: Iterable | None,
    user_fn: Callable[[set, Sequence, int], float],
) -> float:
    pop = list(users) if users is not None else list(predictions.keys())
    if not pop:
        return 0.0
    per_user_scores = [
        user_fn(ground_truth.get(u, set()), predictions.get(u, []), k) for u in pop
    ]
    return float(np.mean(per_user_scores))


# ---------------------------------------------------------------------------
# Public mean-over-users metrics (consistent signature)
# ---------------------------------------------------------------------------

def mapk(
    predictions: Mapping,
    ground_truth: Mapping,
    k: int = 12,
    users: Iterable | None = None,
) -> float:
    """Mean average precision @ k.

    Pass `users=` to control the population averaged over (e.g. all customers
    for Kaggle-style MAP, only buyers for "active-user" MAP).
    """
    return _mean_over_users(predictions, ground_truth, k, users, apk)


def ndcg_at_k(predictions: Mapping[str, Sequence], ground_truth: Mapping[str, set],
              k: int = 12, users: Iterable | None = None) -> float:
    """Mean Normalized DCG @ k (binary relevance)."""
    return _mean_over_users(predictions, ground_truth, k, users, _ndcg_user)


def precision_at_k(predictions: Mapping[str, Sequence], ground_truth: Mapping[str, set],
                   k: int = 12, users: Iterable | None = None) -> float:
    """Mean precision @ k."""
    return _mean_over_users(predictions, ground_truth, k, users, _precision_user)


def recall_at_k(predictions: Mapping[str, Sequence], ground_truth: Mapping[str, set],
                k: int = 12, users: Iterable | None = None) -> float:
    """Mean recall @ k. Default population = users with at least one relevant item
    (matches the "recall over buyers" convention)."""
    if users is None:
        users = [u for u, actual in ground_truth.items() if actual]
    return _mean_over_users(predictions, ground_truth, k, users, _recall_user)


def mrr_at_k(predictions: Mapping[str, Sequence], ground_truth: Mapping[str, set],
             k: int = 12, users: Iterable | None = None) -> float:
    """Mean Reciprocal Rank @ k. 0 contributed by users with no hit in top-k."""
    return _mean_over_users(predictions, ground_truth, k, users, _mrr_user)


def hit_rate_at_k(predictions: Mapping[str, Sequence], ground_truth: Mapping[str, set],
                  k: int = 12, users: Iterable | None = None) -> float:
    """Fraction of users with AT LEAST ONE relevant item in their top-k.

    Per-user binary (hit / no-hit), then averaged. Different from `hit_rate`,
    which is set-based over `(user, item)` pairs — see below.
    """
    pop = list(users) if users is not None else list(predictions.keys())
    if not pop:
        return 0.0
    n_hit = sum(
        1 for u in pop
        if set(predictions.get(u, [])[:k]) & ground_truth.get(u, set())
    )
    return n_hit / len(pop)


# ---------------------------------------------------------------------------
# Retrieval-stage metric (different shape — set-based)
# ---------------------------------------------------------------------------

def hit_rate(candidate_pairs: set, ground_truth_pairs: set) -> float:
    """Fraction of true (user, item) purchases present in the candidate set.

    Different shape from `hit_rate_at_k`: works on sets of (user, item) tuples
    BEFORE the top-k recommendation list exists. Used to measure the
    retrieval-stage ceiling — the reranker can never recommend a pair retrieval
    didn't surface.
    """
    if not ground_truth_pairs:
        return 0.0
    return len(candidate_pairs & ground_truth_pairs) / len(ground_truth_pairs)
