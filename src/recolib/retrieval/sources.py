"""Concrete candidate-retrieval strategies (ported from the validated pipeline).

Each returns (user, item, rank, source). `rank` is per-user, 1 = best. Column
names come from `context.schema`.
"""
from __future__ import annotations

from datetime import timedelta
from functools import reduce

from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from .base import CandidateSource, RetrievalContext


def _rank_top_n(df: DataFrame, partition, order, top_n: int) -> DataFrame:
    """Add a per-`partition` rank by `order` and keep the top `top_n`."""
    w = Window.partitionBy(*partition).orderBy(*order)
    ranked = df.withColumn("rank", F.row_number().over(w)).filter(F.col("rank") <= top_n)
    return ranked


class Repurchase(CandidateSource):
    """Items a user bought before, ranked by purchase count (lifetime if history given)."""
    name = "repurchase"

    def generate(self, transactions, context):
        s = context.schema
        src = context.full_history if context.full_history is not None else transactions
        counts = src.groupBy(s.user, s.item).agg(F.count("*").alias("cnt"))
        ranked = _rank_top_n(counts, [s.user], [F.col("cnt").desc(), F.col(s.item)], self.top_n)
        output = self._tag(ranked.select(s.user, s.item, "rank"))
        return output


class Popularity(CandidateSource):
    """Global top-N most-purchased items in the last `weeks`; given to every customer."""
    name = "popularity"

    def __init__(self, top_n: int = 100, weeks: int = 1):
        super().__init__(top_n)
        self.weeks = weeks

    def generate(self, transactions, context):
        s = context.schema
        cutoff = context.train_end - timedelta(weeks=self.weeks)
        w = Window.orderBy(F.col("n").desc(), F.col(s.item))
        top = (
            transactions.filter(F.col(s.date) > F.lit(cutoff))
            .groupBy(s.item).agg(F.count("*").alias("n"))
            .withColumn("rank", F.row_number().over(w))
            .filter(F.col("rank") <= self.top_n)
            .select(s.item, "rank")
        )
        candidates = context.customers.select(s.user).crossJoin(top).select(s.user, s.item, "rank")
        output = self._tag(candidates)
        return output


class ProductCode(CandidateSource):
    """Variants (same product_code) of items the user recently bought, ranked by item popularity."""
    name = "product_code"

    def __init__(self, top_n: int = 30, seed_weeks: int = 4):
        super().__init__(top_n)
        self.seed_weeks = seed_weeks

    def generate(self, transactions, context):
        s = context.schema
        seed_cutoff = context.train_end - timedelta(weeks=self.seed_weeks)
        art_code = context.articles.select(s.item, "product_code")
        user_codes = (
            transactions.filter(F.col(s.date) > F.lit(seed_cutoff)).select(s.user, s.item).distinct()
            .join(art_code, on=s.item, how="inner").select(s.user, "product_code").distinct()
        )
        item_pop = transactions.groupBy(s.item).agg(F.count("*").alias("pop"))
        variants = (
            context.articles.select("product_code", F.col(s.item).alias("variant"))
            .join(item_pop.withColumnRenamed(s.item, "variant"), on="variant", how="left")
            .fillna({"pop": 0})
        )
        joined = user_codes.join(variants, on="product_code")
        ranked = _rank_top_n(joined, [s.user], [F.col("pop").desc(), F.col("variant")], self.top_n)
        output = self._tag(ranked.select(s.user, F.col("variant").alias(s.item), "rank"))
        return output


class ItemCF(CandidateSource):
    """Item-based collaborative filtering: items co-bought with the user's recent seeds."""
    name = "itemcf"

    def __init__(self, top_n: int = 100, neighbors: int = 50, seed_weeks: int = 4,
                 neighbor_table=None):
        super().__init__(top_n)
        self.neighbors = neighbors
        self.seed_weeks = seed_weeks
        self._neighbors = neighbor_table   # reuse a persisted matrix instead of recomputing

    def neighbor_table(self, transactions, context):
        """Top-`neighbors` co-purchase partners per item.

        Split out because the feature stage needs the same matrix (`ItemCFScore`), and
        recomputing an all-pairs co-occurrence is the most expensive thing in retrieval.
        """
        s = context.schema
        ui = transactions.select(s.user, s.item).distinct()
        cop = (
            ui.alias("a").join(ui.alias("b"), on=s.user)
            .filter(F.col(f"a.{s.item}") != F.col(f"b.{s.item}"))
            .groupBy(F.col(f"a.{s.item}").alias("seed"), F.col(f"b.{s.item}").alias(s.item))
            .agg(F.count("*").alias("cocount"))
        )
        wn = Window.partitionBy("seed").orderBy(F.col("cocount").desc())
        return cop.withColumn("r", F.row_number().over(wn)).filter(F.col("r") <= self.neighbors)

    def generate(self, transactions, context):
        s = context.schema
        nbrs = self._neighbors if self._neighbors is not None else self.neighbor_table(
            transactions, context)
        seed_cutoff = context.train_end - timedelta(weeks=self.seed_weeks)
        seeds = (
            transactions.filter(F.col(s.date) > F.lit(seed_cutoff))
            .select(s.user, F.col(s.item).alias("seed")).distinct()
        )
        scored = seeds.join(nbrs, on="seed").groupBy(s.user, s.item).agg(F.sum("cocount").alias("score"))
        ranked = _rank_top_n(scored, [s.user], [F.col("score").desc(), F.col(s.item)], self.top_n)
        output = self._tag(ranked.select(s.user, s.item, "rank"))
        return output


class ALSGraph(CandidateSource):
    """Graph-embedding candidates via implicit-feedback ALS (matrix factorization)."""
    name = "graph_embedding"

    def __init__(self, top_n: int = 100, rank: int = 64, iters: int = 10,
                 reg: float = 0.1, alpha: float = 40.0, seed: int = 42):
        super().__init__(top_n)
        self.rank, self.iters, self.reg, self.alpha, self.seed = rank, iters, reg, alpha, seed

    def generate(self, transactions, context):
        s = context.schema
        ratings = transactions.select(s.user, s.item).distinct().withColumn("purchased", F.lit(1))
        uix = StringIndexer(inputCol=s.user, outputCol="u").fit(ratings)
        iix = StringIndexer(inputCol=s.item, outputCol="i").fit(ratings)
        idx = iix.transform(uix.transform(ratings)) \
            .withColumn("u", F.col("u").cast("int")).withColumn("i", F.col("i").cast("int"))
        model = ALS(userCol="u", itemCol="i", ratingCol="purchased", rank=self.rank,
                    maxIter=self.iters, regParam=self.reg, alpha=self.alpha,
                    implicitPrefs=True, coldStartStrategy="drop", nonnegative=True,
                    seed=self.seed).fit(idx)
        recs = model.recommendForAllUsers(self.top_n)
        umap = context.spark.createDataFrame(list(enumerate(uix.labels)), ["u", s.user])
        imap = context.spark.createDataFrame(list(enumerate(iix.labels)), ["i", s.item])
        candidates = (
            recs.select("u", F.posexplode("recommendations").alias("pos", "rec"))
            .select("u", F.col("rec.i").alias("i"), (F.col("pos") + 1).alias("rank"))
            .join(F.broadcast(umap), on="u").join(F.broadcast(imap), on="i")
            .select(s.user, s.item, "rank")
        )
        output = self._tag(candidates)
        return output


class SegmentPopularity(CandidateSource):
    """Demographic-popularity candidates with backoff.

    For each customer, returns top items in priority order:
      1. items popular within the customer's (age_group, postal_code) segment
      2. items popular within the customer's age_group
      3. (optional) globally popular items, as a fallback

    Uses only customer demographics — serves brand-new users with no purchase history.
    Requires `context.customers` to have `age` and `postal_code` columns.

    The fallback in step 3 is controlled by `global_fallback`:
      - `"all"` (default): global popularity backfills EVERY customer's tail once
        steps 1+2 are exhausted (the original "coldstart" design).
      - `"no_info_only"`: demographic segments are respected strictly — global is
        added ONLY for customers with NO age AND no postal_code (nothing to
        segment by). Pair with `name="popularity"` to replace both a
        global-popularity source AND a coldstart source in one (the design that
        scored best in production).
      - `"off"`: no global tail at all — pure demographic, no fallback.
    """
    name = "coldstart"

    def __init__(
        self,
        top_n: int = 50,
        weeks: int = 4,
        global_fallback: str = "all",
        name: str | None = None,
    ):
        super().__init__(top_n)
        valid = ("all", "no_info_only", "off")
        if global_fallback not in valid:
            raise ValueError(
                f"global_fallback must be one of {valid}, got {global_fallback!r}"
            )
        self.weeks = weeks
        self.global_fallback = global_fallback
        if name is not None:
            self.name = name

    @staticmethod
    def _age_group(col):
        age_group = (F.when(col < 25, "youth").when(col < 35, "young_adult")
                     .when(col < 45, "adult").when(col < 55, "middle")
                     .when(col.isNotNull(), "senior").otherwise("unknown"))
        return age_group

    def _topk(self, recent, keys, s):
        w = Window.partitionBy(*keys).orderBy(F.col("n").desc(), F.col(s.item))
        top = (recent.groupBy(*keys, s.item).agg(F.count("*").alias("n"))
               .withColumn("rk", F.row_number().over(w)).filter(F.col("rk") <= self.top_n))
        return top

    def generate(self, transactions, context):
        s = context.schema
        cust = (
            context.customers.select(s.user, F.col("age").cast("double").alias("age"), "postal_code")
            .withColumn("age_group", self._age_group(F.col("age")))
            .fillna({"postal_code": "NA"}).select(s.user, "age_group", "postal_code")
        )
        cutoff = context.train_end - timedelta(weeks=self.weeks)
        recent = transactions.filter(F.col(s.date) > F.lit(cutoff)).join(cust, on=s.user, how="left")

        l1 = cust.join(self._topk(recent, ["age_group", "postal_code"], s), on=["age_group", "postal_code"]) \
            .select(s.user, s.item, (F.lit(0) * 1_000_000 + F.col("rk")).alias("pri"))
        l2 = cust.join(self._topk(recent, ["age_group"], s), on="age_group") \
            .select(s.user, s.item, (F.lit(1) * 1_000_000 + F.col("rk")).alias("pri"))
        levels = [l1, l2]
        if self.global_fallback != "off":
            gw = Window.orderBy(F.col("n").desc(), F.col(s.item))
            g = (recent.groupBy(s.item).agg(F.count("*").alias("n"))
                 .withColumn("rk", F.row_number().over(gw)).filter(F.col("rk") <= self.top_n))
            if self.global_fallback == "no_info_only":
                # global only for users with no age AND no postal
                raw = context.customers.select(s.user, F.col("age"), "postal_code")
                target = raw.filter(
                    F.col("age").isNull()
                    & (F.col("postal_code").isNull() | (F.col("postal_code") == "NA"))
                ).select(s.user)
            else:  # "all"
                target = cust.select(s.user)
            levels.append(target.crossJoin(F.broadcast(g)).select(
                s.user, s.item, (F.lit(2) * 1_000_000 + F.col("rk")).alias("pri")))

        merged = reduce(lambda a, b: a.unionByName(b), levels)
        merged = merged.groupBy(s.user, s.item).agg(F.min("pri").alias("pri"))
        ranked = _rank_top_n(merged, [s.user], [F.col("pri")], self.top_n)
        output = self._tag(ranked.select(s.user, s.item, "rank"))
        return output
