"""Tests for recolib.data — loader, preprocess, splitter (Path B: column names as parameters)."""
from __future__ import annotations

from datetime import date, timedelta

import pytest

from recolib.data.loader import DataLoader
from recolib.data.preprocess import (
    dedup_transactions,
    filter_date_range,
    filter_min_purchases,
    add_week,
)
from recolib.data.splitter import Fold, TimeSeriesSplitter, RandomSplitter


def _transactions(spark, rows, cols=("customer_id", "article_id", "t_dat")):
    return spark.createDataFrame(rows, list(cols))


class TestLoader:
    def test_infer_fmt(self):
        assert DataLoader._infer_fmt("foo.csv") == "csv"
        assert DataLoader._infer_fmt("/a/b/c.json") == "json"
        assert DataLoader._infer_fmt("path/x.parquet") == "parquet"
        assert DataLoader._infer_fmt("/data/some_table") == "parquet"   # no ext -> parquet

    def test_resolve_with_base_dir(self, spark):
        loader = DataLoader(spark, base_dir="/data")
        assert loader._resolve("file.csv") == "/data/file.csv"
        assert loader._resolve("/abs/path.csv") == "/abs/path.csv"      # absolute path passes through

    def test_resolve_without_base_dir(self, spark):
        loader = DataLoader(spark)
        assert loader._resolve("file.csv") == "file.csv"

    def test_load_csv(self, spark, tmp_path):
        csv = tmp_path / "data.csv"
        csv.write_text("a,b\n1,2\n3,4\n")
        df = DataLoader(spark).load(str(csv))
        rows = sorted((r["a"], r["b"]) for r in df.collect())
        assert rows == [("1", "2"), ("3", "4")]

    def test_load_parquet(self, spark, tmp_path):
        out = tmp_path / "tbl"
        spark.createDataFrame([(1, "x"), (2, "y")], ["i", "s"]).write.parquet(str(out))
        df = DataLoader(spark).load(str(out))
        rows = sorted((r["i"], r["s"]) for r in df.collect())
        assert rows == [(1, "x"), (2, "y")]

    def test_apply_casts_date_and_double(self, spark):
        df = spark.createDataFrame([("2020-01-15", "3.5"), ("2020-02-20", "9.0")], ["d", "x"])
        out = DataLoader.apply_casts(df, casts={"d": "date", "x": "double"})
        rows = sorted(out.collect(), key=lambda r: r["d"])
        assert rows[0]["d"] == date(2020, 1, 15)
        assert rows[0]["x"] == 3.5

    def test_apply_casts_empty_is_noop(self, spark):
        df = spark.createDataFrame([(1,), (2,)], ["x"])
        out = DataLoader.apply_casts(df, casts={})
        assert out.columns == df.columns and out.count() == 2


class TestPreprocess:
    def test_dedup_basic_drops_dupes_and_extra_cols(self, spark):
        df = spark.createDataFrame(
            [("u1", "a1", date(2020, 1, 1), 5.0),
             ("u1", "a1", date(2020, 1, 1), 5.0),   # exact dup
             ("u1", "a2", date(2020, 1, 1), 7.0)],
            ["customer_id", "article_id", "t_dat", "price"],
        )
        out = dedup_transactions(df).orderBy("article_id")
        rows = [(r["customer_id"], r["article_id"]) for r in out.collect()]
        assert rows == [("u1", "a1"), ("u1", "a2")]
        assert set(out.columns) == {"customer_id", "article_id", "t_dat"}     # 'price' dropped

    def test_dedup_with_extras_keeps_first_nonnull(self, spark):
        df = spark.createDataFrame(
            [("u1", "a1", date(2020, 1, 1), None),
             ("u1", "a1", date(2020, 1, 1), 5.0)],
            ["customer_id", "article_id", "t_dat", "price"],
        )
        out = dedup_transactions(df, extra_first_cols=["price"])
        assert out.count() == 1
        assert out.first()["price"] == 5.0       # the None was skipped

    def test_filter_date_range_both_bounds_inclusive(self, spark):
        df = _transactions(spark, [
            ("u1", "a1", date(2020, 1, 1)),
            ("u1", "a1", date(2020, 3, 1)),    # exactly on start
            ("u1", "a1", date(2020, 6, 1)),
            ("u1", "a1", date(2020, 9, 1)),    # exactly on end
            ("u1", "a1", date(2021, 1, 1)),
        ])
        out = filter_date_range(df, start=date(2020, 3, 1), end=date(2020, 9, 1))
        assert out.count() == 3

    def test_filter_date_range_open_bound(self, spark):
        df = _transactions(spark, [
            ("u1", "a1", date(2020, 1, 1)),
            ("u1", "a1", date(2020, 6, 1)),
        ])
        assert filter_date_range(df, end=date(2020, 3, 1)).count() == 1   # start=None
        assert filter_date_range(df, start=date(2020, 4, 1)).count() == 1 # end=None

    def test_filter_min_purchases_keeps_only_high_volume(self, spark):
        df = _transactions(spark, [
            ("u1", "a1", date(2020, 1, 1)),
            ("u1", "a2", date(2020, 1, 2)),
            ("u2", "a1", date(2020, 1, 3)),    # u2 has only 1 -> dropped at min_n=2
        ])
        out = filter_min_purchases(df, min_n=2)
        users = {r["customer_id"] for r in out.collect()}
        assert users == {"u1"}

    def test_add_week_indexes_from_earliest(self, spark):
        df = _transactions(spark, [
            ("u1", "a1", date(2020, 1, 1)),       # week 0
            ("u1", "a1", date(2020, 1, 8)),       # week 1
            ("u1", "a1", date(2020, 1, 22)),      # week 3
        ])
        out = add_week(df).orderBy("t_dat")
        assert [r["week"] for r in out.collect()] == [0, 1, 3]


class TestSplitter:
    def test_fold_is_submission_when_val_none(self):
        f = Fold(name="submission", train=None, val=None)
        assert f.is_submission

    def test_time_series_basic_shape_and_dates(self, spark):
        # 80 days of rows up to and including window_end 2020-09-22
        rows = [("u1", "a1", date(2020, 9, 22) - timedelta(days=d)) for d in range(80)]
        df = _transactions(spark, rows)
        splitter = TimeSeriesSplitter(window_end="2020-09-22", train_weeks=4, val_weeks=1,
                                      n_folds=2, with_submission=True)
        folds = splitter.split(df)
        assert [f.name for f in folds] == ["fold_0", "fold_1", "submission"]
        # fold_0: val = the last week ending window_end
        assert folds[0].meta["val_end"] == date(2020, 9, 22)
        assert folds[0].meta["train_end"] == date(2020, 9, 15)
        assert folds[0].meta["train_end"] < folds[0].meta["val_start"]   # leakage-safe
        # submission fold: val is None
        assert folds[-1].is_submission

    def test_random_splitter_by_row(self, spark):
        df = _transactions(spark, [(f"u{i}", "a1", date(2020, 1, 1)) for i in range(100)])
        with pytest.warns(UserWarning):
            folds = RandomSplitter(val_frac=0.2, test_frac=0.0, by="row", seed=1).split(df)
        assert len(folds) == 1
        f = folds[0]
        assert f.train.count() + f.val.count() == 100
        assert f.test is None

    def test_random_splitter_by_user_disjoint(self, spark):
        df = _transactions(spark, [(f"u{i}", "a1", date(2020, 1, 1)) for i in range(50)])
        with pytest.warns(UserWarning):
            folds = RandomSplitter(val_frac=0.3, test_frac=0.0, by="user", seed=2).split(df)
        f = folds[0]
        train_users = {r["customer_id"] for r in f.train.collect()}
        val_users   = {r["customer_id"] for r in f.val.collect()}
        assert train_users.isdisjoint(val_users)    # by-user => no user in both splits
