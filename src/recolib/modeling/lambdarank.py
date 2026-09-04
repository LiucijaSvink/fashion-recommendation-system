"""LightGBM LambdaRank reranker — operates on a pandas feature table.

Collect the (downsampled) feature table from Spark, then::

    model = LambdaRanker().fit(train_pdf)
    recs  = model.recommend(eval_pdf, k=12)
    print(rl.metrics.mapk(recs, ground_truth, k=12))

Column names come from the `schema` passed at construction.

Supports:
  * native LightGBM categorical splits (`categorical_features=`)
  * early stopping via an optional eval frame + `callbacks=[lgb.early_stopping(N)]`
  * popularity backfill for users with fewer than k scored candidates
  * gain- or split-based feature importance
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..schema import DEFAULT_SCHEMA, Schema


class LambdaRanker:
    """LightGBM LambdaRank wrapper with categorical, early-stopping, and backfill support."""

    DEFAULT_PARAMS: dict = dict(
        objective="lambdarank", metric="ndcg",
        n_estimators=400, learning_rate=0.05,
        num_leaves=63, min_child_samples=50,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, random_state=42, n_jobs=-1,
    )

    def __init__(
        self,
        feature_cols: list[str] | None = None,
        categorical_features: list[str] | None = None,
        schema: Schema = DEFAULT_SCHEMA,
        **lgbm_params,
    ):
        self.feature_cols = feature_cols
        self.categorical_features = categorical_features
        self.schema = schema
        self.params = {**self.DEFAULT_PARAMS, **lgbm_params}
        self.model: lgb.LGBMRanker | None = None

    def _resolve_features(self, pdf: pd.DataFrame) -> list[str]:
        """Pick feature columns from the pdf, excluding keys/labels/fold tag."""
        if self.feature_cols:
            return self.feature_cols
        skip = {self.schema.user, self.schema.item, "label", "_fold"}
        feature_cols = [c for c in pdf.columns if c not in skip]
        return feature_cols

    def _prepare_X(self, pdf: pd.DataFrame) -> pd.DataFrame:
        """Cast to float32 and fillna(0). Categorical columns stay numeric for LightGBM."""
        X = pdf[self.feature_cols].astype("float32").fillna(0)
        return X

    def fit(
        self,
        train_pdf: pd.DataFrame,
        group_cols: list[str] | None = None,
        eval_pdf: pd.DataFrame | None = None,
        eval_group_cols: list[str] | None = None,
        callbacks: list | None = None,
    ) -> "LambdaRanker":
        """Fit on a pandas table with `label` and a ranking-group key.

        Parameters
        ----------
        train_pdf       : feature table; must contain `label`
        group_cols      : ranking-group key (default `[schema.user]`; use
                          `[_fold, schema.user]` for multi-fold training)
        eval_pdf        : optional held-out frame for early stopping
        eval_group_cols : group key for `eval_pdf` (defaults to `group_cols`)
        callbacks       : LightGBM callbacks — e.g. `[lgb.early_stopping(50),
                          lgb.log_evaluation(25)]`
        """
        group_cols = group_cols or [self.schema.user]
        self.feature_cols = self._resolve_features(train_pdf)

        train_pdf = train_pdf.sort_values(group_cols).reset_index(drop=True)
        groups = train_pdf.groupby(group_cols, sort=False).size().values
        X = self._prepare_X(train_pdf)
        y = train_pdf["label"].values

        fit_kwargs: dict = {"group": groups}
        if self.categorical_features:
            fit_kwargs["categorical_feature"] = self.categorical_features
        if eval_pdf is not None:
            eval_group_cols = eval_group_cols or group_cols
            eval_pdf = eval_pdf.sort_values(eval_group_cols).reset_index(drop=True)
            eval_groups = eval_pdf.groupby(eval_group_cols, sort=False).size().values
            fit_kwargs["eval_set"] = [(self._prepare_X(eval_pdf), eval_pdf["label"].values)]
            fit_kwargs["eval_group"] = [eval_groups]
        if callbacks:
            fit_kwargs["callbacks"] = callbacks

        self.model = lgb.LGBMRanker(**self.params)
        self.model.fit(X, y, **fit_kwargs)
        return self

    def score(self, pdf: pd.DataFrame) -> np.ndarray:
        """Predict per-row LambdaRank scores. Higher = more relevant."""
        if self.model is None:
            raise RuntimeError("call fit() first")
        return self.model.predict(self._prepare_X(pdf))

    def recommend(
        self,
        pdf: pd.DataFrame,
        k: int = 12,
        backfill: dict | None = None,
    ) -> dict[str, list]:
        """Return `{user: [top-k items]}` from scored candidates.

        Parameters
        ----------
        pdf      : scored feature table (must have schema.user, schema.item)
        k        : list length to return per user
        backfill : optional `{user: [fallback items]}` (typically a popularity
                   list) used to pad users with fewer than k scored candidates,
                   and to surface users absent from `pdf` entirely. Without it,
                   cold users silently drop out of the result.

        No mutation of `pdf` — a small (user, item, score) frame is built
        internally for sorting, so any existing columns on `pdf` are safe.
        """
        if self.model is None:
            raise RuntimeError("call fit() first")
        s = self.schema
        scores = self.score(pdf)
        sort_df = pd.DataFrame({
            s.user: pdf[s.user].values,
            s.item: pdf[s.item].values,
            "_score": scores,
        })
        # `.head(k)` is a vectorised cumcount, so the per-group Python call happens
        # on k rows per user instead of on every scored candidate
        top_k = (
            sort_df.sort_values([s.user, "_score"], ascending=[True, False])
            .groupby(s.user, sort=False)
            .head(k)
        )
        recs = top_k.groupby(s.user, sort=False)[s.item].agg(list).to_dict()
        if backfill:
            for user, fallback in backfill.items():
                cur = recs.get(user, [])
                if len(cur) < k:
                    seen = set(cur)
                    extra = [it for it in fallback if it not in seen]
                    recs[user] = (cur + extra)[:k]
        return recs

    def feature_importance(self, importance_type: str = "gain") -> pd.DataFrame:
        """Sorted feature importance. `importance_type` is "gain" or "split"."""
        if self.model is None:
            raise RuntimeError("call fit() first")
        importance = (
            pd.DataFrame({
                "feature": self.feature_cols,
                importance_type: self.model.booster_.feature_importance(importance_type),
            })
            .sort_values(importance_type, ascending=False)
            .reset_index(drop=True)
        )
        return importance
