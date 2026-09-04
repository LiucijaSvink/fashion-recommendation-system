"""EDA analysis + plotting helpers for `eda.ipynb`.

`compute_*` do the Spark work and return small pandas frames; `plot_*` draw them;
`analyze_*` orchestrate compute + plot + printed summary. The fold-based helpers
(`make_fold`, `evaluate_source`, `consistency`, `price_tier`, `top_index`) support
the Section-4 "what predicts next week" analyses.

Kept out of `recolib`: these are notebook-specific analysis and matplotlib helpers,
not part of the reusable recommender library.
"""
from __future__ import annotations

import os
from datetime import timedelta

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Window
from scipy.stats import spearmanr

from recolib.data import TimeSeriesSplitter

# H&M column names, shared by the fold-based analyses
CUSTOMER, ITEM, DATE = "customer_id", "article_id", "t_dat"

__all__ = [
    "CUSTOMER", "ITEM", "DATE", "chart_title",
    "compute_transaction_patterns", "plot_transaction_patterns", "analyze_transaction_patterns",
    "compute_interaction_distribution", "plot_interaction_distribution", "analyze_interaction_distribution",
    "compute_item_decay", "plot_item_decay",
    "compute_demand_concentration", "plot_demand_concentration",
    "compute_repurchase", "plot_repurchase",
    "compute_user_activity_segments", "plot_user_activity_segments",
    "make_fold", "evaluate_source", "consistency", "price_tier", "top_index",
]


def chart_title(fig, name):
    fig.suptitle(name, fontsize=12, fontweight="bold")


def compute_transaction_patterns(df: DataFrame) -> pd.DataFrame:
    """
    Computes weekly transaction patterns in Spark.
    Outliers flagged as weeks > 2 std from mean.
    Returns small pandas DataFrame (~100 rows) safe to collect.
    """
    weekly = (
        df.withColumn("week", F.date_trunc("week", F.col("t_dat")))
          .groupBy("week")
          .agg(
              F.count("*").alias("n_transactions"),
              F.countDistinct("customer_id").alias("n_active_users"),
              F.countDistinct("article_id").alias("n_distinct_items"),
              F.avg("price").alias("avg_price"),
          )
          .orderBy("week")
    )

    stats = weekly.agg(
        F.mean("n_transactions").alias("mean"),
        F.stddev("n_transactions").alias("std"),
    ).collect()[0]

    weekly = (
        weekly
        .withColumn("is_outlier",
            (F.col("n_transactions") > stats["mean"] + 2 * stats["std"]) |
            (F.col("n_transactions") < stats["mean"] - 2 * stats["std"])
        )
        .toPandas()   # safe — only ~100 rows
    )

    weekly["week"] = pd.to_datetime(weekly["week"])
    return weekly


def plot_transaction_patterns(weekly: pd.DataFrame) -> plt.Figure:
    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    chart_title(fig, "Transaction patterns over time")

    metrics = [
        ("n_transactions",   "steelblue",  "Transactions", "Weekly Transaction Volume"),
        ("n_active_users",   "darkorange", "Users",        "Weekly Active Users"),
        ("n_distinct_items", "seagreen",   "Items",        "Weekly Distinct Items Purchased"),
        ("avg_price",        "firebrick",  "Avg Price",    "Weekly Average Price"),
    ]

    for ax, (col, color, ylabel, title) in zip(axes, metrics):
        mean = weekly[col].mean()
        std  = weekly[col].std()

        ax.plot(weekly["week"], weekly[col], color=color, linewidth=1.5)

        ax.axhline(mean, color="gray", linestyle="--", alpha=0.7, linewidth=1, label="mean")

        ax.axhline(mean + 2 * std, color="red", linestyle="--", alpha=0.7, linewidth=1, label="+2σ")
        ax.axhline(mean - 2 * std, color="red", linestyle="--", alpha=0.7, linewidth=1, label="-2σ")

        ax.set_ylabel(ylabel)
        ax.set_title(title, fontweight="bold")
        ax.xaxis.set_major_locator(ticker.MaxNLocator(12))
        ax.tick_params(axis="x", rotation=45)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)

    plt.tight_layout()
    return fig


def analyze_transaction_patterns(df: DataFrame) -> plt.Figure:
    """Orchestrates compute + plot + prints summary stats."""

    weekly = compute_transaction_patterns(df)
    fig    = plot_transaction_patterns(weekly)

    outliers = weekly[weekly["is_outlier"]]

    print("\n--- Summary ---")
    print(f"Date range    : {weekly['week'].min().date()} → {weekly['week'].max().date()}")
    print(f"Total weeks   : {len(weekly)}")
    print(f"Peak volume   : {weekly.loc[weekly['n_transactions'].idxmax(), 'week'].date()} "
          f"({weekly['n_transactions'].max():,} transactions)")
    print(f"Lowest volume : {weekly.loc[weekly['n_transactions'].idxmin(), 'week'].date()} "
          f"({weekly['n_transactions'].min():,} transactions)")
    print(f"Outlier weeks : {len(outliers)}")
    for _, row in outliers.iterrows():
        print(f"  {row['week'].date()} — {row['n_transactions']:,} transactions")

    return fig


def _parquet_exists(path):
    return os.path.exists(path) and len(os.listdir(path)) > 0


def _print_section(title):
    print("\n" + "=" * 65)
    print(f"  {title}")
    print("=" * 65)


def _style_ax(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.tick_params(axis="both", labelsize=8)


def compute_interaction_distribution(transactions, save_path):
    paths = {
        "summary"      : os.path.join(save_path, "summary"),
        "user_buckets" : os.path.join(save_path, "user_buckets"),
        "item_buckets" : os.path.join(save_path, "item_buckets"),
    }

    if all(_parquet_exists(p) for p in paths.values()):
        print(f"  ✓ Found cached data at {save_path}, skipping recomputation.")
        return paths

    os.makedirs(save_path, exist_ok=True)

    summary = transactions.agg(
        F.count("*").alias("n_transactions"),
        F.countDistinct("customer_id").alias("n_users"),
        F.countDistinct("article_id").alias("n_items"),
    ).withColumn(
        "sparsity",
        F.lit(1.0) - (F.col("n_transactions") / (F.col("n_users") * F.col("n_items")))
    )
    summary.write.mode("overwrite").parquet(paths["summary"])

    user_buckets = (
        transactions.groupBy("customer_id")
          .agg(F.count("*").alias("n_interactions"))
          .withColumn("log_bucket", F.floor(F.log2(F.col("n_interactions"))).cast("int"))
          .groupBy("log_bucket")
          .agg(F.count("*").alias("n_users"))
          .orderBy("log_bucket")
          .withColumn("bucket_label",
              F.concat(
                  F.pow(F.lit(2), F.col("log_bucket")).cast("int").cast("string"),
                  F.lit(" - "),
                  (F.pow(F.lit(2), F.col("log_bucket") + 1).cast("int") - 1).cast("string")
              ))
    )
    user_buckets.write.mode("overwrite").parquet(paths["user_buckets"])

    item_buckets = (
        transactions.groupBy("article_id")
          .agg(F.count("*").alias("n_interactions"))
          .withColumn("log_bucket", F.floor(F.log2(F.col("n_interactions"))).cast("int"))
          .groupBy("log_bucket")
          .agg(F.count("*").alias("n_items"))
          .orderBy("log_bucket")
          .withColumn("bucket_label",
              F.concat(
                  F.pow(F.lit(2), F.col("log_bucket")).cast("int").cast("string"),
                  F.lit(" - "),
                  (F.pow(F.lit(2), F.col("log_bucket") + 1).cast("int") - 1).cast("string")
              ))
    )
    item_buckets.write.mode("overwrite").parquet(paths["item_buckets"])

    print(f"  ✓ Saved to {save_path}")
    return paths


def plot_interaction_distribution(paths, spark):
    summary      = spark.read.parquet(paths["summary"]).toPandas().iloc[0]
    user_buckets = spark.read.parquet(paths["user_buckets"]).toPandas().sort_values("log_bucket")
    item_buckets = spark.read.parquet(paths["item_buckets"]).toPandas().sort_values("log_bucket")

    cold_user_pct = (
        user_buckets[user_buckets["log_bucket"] == 0]["n_users"].sum()
        / user_buckets["n_users"].sum() * 100
    )
    cold_item_pct = (
        item_buckets[item_buckets["log_bucket"] == 0]["n_items"].sum()
        / item_buckets["n_items"].sum() * 100
    )

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    chart_title(fig, "Interaction matrix & sparsity")

    axes[0].bar(user_buckets["bucket_label"], user_buckets["n_users"],
                color="steelblue", edgecolor="white", alpha=0.85)
    axes[0].tick_params(axis="x", rotation=45)
    _style_ax(axes[0], "User Interaction Distribution", "log2 bucket", "# users")

    axes[1].bar(item_buckets["bucket_label"], item_buckets["n_items"],
                color="darkorange", edgecolor="white", alpha=0.85)
    axes[1].tick_params(axis="x", rotation=45)
    _style_ax(axes[1], "Item Interaction Distribution", "log2 bucket", "# items")

    labels = ["Sparsity (%)", "Cold users (%)", "Cold items (%)"]
    values = [summary["sparsity"] * 100, cold_user_pct, cold_item_pct]
    colors = ["steelblue", "darkorange", "seagreen"]
    bars   = axes[2].barh(labels, values, color=colors, edgecolor="white", alpha=0.85)
    axes[2].set_xlim(0, 100)
    # 2 decimals throughout — sparsity rounds to a misleading "100.0%" at 1
    for bar, val in zip(bars, values):
        axes[2].text(val + 1, bar.get_y() + bar.get_height() / 2,
                     f"{val:.2f}%", va="center", fontsize=9)
    _style_ax(axes[2], "Key Sparsity Metrics", "%")

    plt.tight_layout()
    return fig


def analyze_interaction_distribution(transactions, save_path, spark):
    _print_section("Interaction matrix & sparsity")

    paths        = compute_interaction_distribution(transactions, save_path)
    fig          = plot_interaction_distribution(paths, spark)
    summary      = spark.read.parquet(paths["summary"]).toPandas().iloc[0]
    user_buckets = spark.read.parquet(paths["user_buckets"]).toPandas()
    item_buckets = spark.read.parquet(paths["item_buckets"]).toPandas()

    cold_user_pct = (
        user_buckets[user_buckets["log_bucket"] == 0]["n_users"].sum()
        / user_buckets["n_users"].sum() * 100
    )
    cold_item_pct = (
        item_buckets[item_buckets["log_bucket"] == 0]["n_items"].sum()
        / item_buckets["n_items"].sum() * 100
    )

    print(f"  n_transactions : {summary['n_transactions']:,.0f}")
    print(f"  n_users        : {summary['n_users']:,.0f}")
    print(f"  n_items        : {summary['n_items']:,.0f}")
    print(f"  sparsity       : {summary['sparsity'] * 100:.4f}%")
    print(f"  cold users     : {cold_user_pct:.1f}% have exactly 1 interaction")
    print(f"  cold items     : {cold_item_pct:.1f}% have exactly 1 interaction")

    return fig


def compute_item_decay(df: DataFrame, date_col: str = "t_dat", item_col: str = "article_id", max_weeks: int = None) -> dict:
    """
    Computes item decay metrics in Spark.

    Parameters
    ----------
    max_weeks : int, optional
        Cap the survival curve and volume curve at this many weeks. If None, show all weeks.

    Returns
    -------
    dict with keys:
        "lifespan_dist"        : pd.DataFrame — lifespan distribution
        "survival_curve"       : pd.DataFrame — % items still active per week offset
        "volume_by_week_offset": pd.DataFrame — % of total purchases per week offset
                                                 (week 0 = launch week, week 1 = next week, ...)
    """

    item_activity = (
        df.withColumn("week", F.date_trunc("week", F.col(date_col)))
          .groupBy(item_col)
          .agg(
              F.min("week").alias("first_week"),
              F.max("week").alias("last_week"),
          )
          .withColumn(
              "lifespan_weeks",
              (F.datediff(F.col("last_week"), F.col("first_week")) / 7).cast("int")
          )
    )

    lifespan_dist = (
        item_activity
        .groupBy("lifespan_weeks")
        .agg(F.count("*").alias("n_items"))
        .orderBy("lifespan_weeks")
        .toPandas()
    )

    item_weekly = (
        df.withColumn("week", F.date_trunc("week", F.col(date_col)))
          .groupBy(item_col, "week")
          .agg(F.count("*").alias("n_purchases"))
    )

    item_first_week = item_activity.select(item_col, "first_week")

    survival = (
        item_weekly
        .join(item_first_week, on=item_col, how="left")
        .withColumn(
            "week_offset",
            (F.datediff(F.col("week"), F.col("first_week")) / 7).cast("int")
        )
    )

    n_total_items = item_activity.count()

    survival_curve = (
        survival
        .groupBy("week_offset")
        .agg(F.countDistinct(item_col).alias("n_active_items"))
        .orderBy("week_offset")
        .withColumn("pct_surviving", F.col("n_active_items") / F.lit(n_total_items) * 100)
    )

    n_total_purchases = df.count()

    volume_by_week_offset = (
        survival
        .groupBy("week_offset")
        .agg(F.sum("n_purchases").alias("n_purchases"))
        .orderBy("week_offset")
        .withColumn("pct_volume", F.col("n_purchases") / F.lit(n_total_purchases) * 100)
    )

    if max_weeks is not None:
        survival_curve = survival_curve.filter(F.col("week_offset") <= max_weeks)
        volume_by_week_offset = volume_by_week_offset.filter(F.col("week_offset") <= max_weeks)

    return {
        "lifespan_dist"        : lifespan_dist,
        "survival_curve"       : survival_curve.toPandas(),
        "volume_by_week_offset": volume_by_week_offset.toPandas(),
    }


def plot_item_decay(data: dict) -> plt.Figure:
    """
    Plots item decay metrics.

    Panel 1: Item lifespan distribution — how long do items stay active?
    Panel 2: Survival curve — % of items still active at week N
    Panel 3: Cumulative volume curve — % of total purchases captured by week N since launch
    """
    lifespan_dist         = data["lifespan_dist"]
    survival_curve        = data["survival_curve"]
    volume_by_week_offset = data["volume_by_week_offset"]

    fig, axes = plt.subplots(1, 3, figsize=(20, 5))
    chart_title(fig, "Item decay / lifecycle")

    axes[0].bar(
        lifespan_dist["lifespan_weeks"],
        lifespan_dist["n_items"],
        color="steelblue", edgecolor="white", alpha=0.85, width=1,
    )
    median_lifespan = (lifespan_dist["lifespan_weeks"] * lifespan_dist["n_items"]).sum() / lifespan_dist["n_items"].sum()
    axes[0].axvline(median_lifespan, color="firebrick", linestyle="--",
                    linewidth=1.5, label=f"Mean: {median_lifespan:.1f} weeks")
    axes[0].set_xlabel("lifespan (weeks)")
    axes[0].set_ylabel("# items")
    axes[0].set_title("Item Lifespan Distribution")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(
        survival_curve["week_offset"],
        survival_curve["pct_surviving"],
        color="darkorange", linewidth=2,
    )
    axes[1].fill_between(
        survival_curve["week_offset"],
        survival_curve["pct_surviving"],
        alpha=0.15, color="darkorange",
    )
    for pct, color in [(50, "firebrick"), (20, "seagreen")]:
        row = survival_curve[survival_curve["pct_surviving"] <= pct]
        if not row.empty:
            week = row.iloc[0]["week_offset"]
            axes[1].axhline(pct, color=color, linestyle="--", alpha=0.7,
                            linewidth=1, label=f"{pct}% survive until week {week:.0f}")
            axes[1].axvline(week, color=color, linestyle="--", alpha=0.7, linewidth=1)
    axes[1].set_xlabel("weeks since first purchase")
    axes[1].set_ylabel("% items still active")
    axes[1].set_title("Item Survival Curve")
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.25)

    cum_vol = volume_by_week_offset["pct_volume"].cumsum()
    axes[2].plot(
        volume_by_week_offset["week_offset"],
        cum_vol,
        color="mediumpurple", linewidth=2,
    )
    axes[2].fill_between(
        volume_by_week_offset["week_offset"],
        cum_vol,
        alpha=0.15, color="mediumpurple",
    )
    for pct, color in [(50, "firebrick"), (80, "seagreen"), (95, "steelblue")]:
        row = volume_by_week_offset[cum_vol >= pct]
        if not row.empty:
            week = row.iloc[0]["week_offset"]
            axes[2].axhline(pct, color=color, linestyle="--", alpha=0.7,
                            linewidth=1, label=f"{pct}% of volume by week {week:.0f}")
            axes[2].axvline(week, color=color, linestyle="--", alpha=0.7, linewidth=1)
    axes[2].set_xlabel("weeks since first purchase")
    axes[2].set_ylabel("cumulative % of total purchases")
    axes[2].set_title("Cumulative Purchase Volume Since Launch")
    axes[2].legend(fontsize=9)
    axes[2].grid(True, alpha=0.25)

    plt.tight_layout()

    half_life = survival_curve[survival_curve["pct_surviving"] <= 50]
    vol_80 = volume_by_week_offset[cum_vol >= 80]
    print(f"\n  Item lifecycle:")
    if not half_life.empty:
        print(f"     50% of items go cold after {half_life.iloc[0]['week_offset']:.0f} weeks (survival).")
    if not vol_80.empty:
        print(f"     80% of purchase volume is captured within {vol_80.iloc[0]['week_offset']:.0f} weeks of launch.")

    return fig


def compute_demand_concentration(
    transactions: DataFrame,
    date_col: str = "t_dat",
    item_col: str = "article_id",
    top_n: int = 1000,
) -> dict:
    """
    Computes demand concentration metrics in Spark.

    Parameters
    ----------
    transactions       : transactions Spark DataFrame
    date_col : date column name
    item_col : item column name
    top_n    : number of top items to use for rank correlation

    Returns
    -------
    dict with keys:
        "lorenz"       : pd.DataFrame — item_pct, cum_purchase_pct
        "weekly_ranks" : pd.DataFrame — week, spearman_r
        "summary"      : pd.DataFrame — top1_pct, top10_pct, mean_rank_corr
    """

    item_counts = (
        transactions.groupBy(item_col)
          .agg(F.count("*").alias("n_purchases"))
          .orderBy("n_purchases", ascending=False)
          .toPandas()
    )

    total_purchases = item_counts["n_purchases"].sum()
    n_items = len(item_counts)

    item_counts["cum_purchase_pct"] = item_counts["n_purchases"].cumsum() / total_purchases * 100
    item_counts["item_pct"]         = (item_counts.index + 1) / n_items * 100

    lorenz = item_counts[["item_pct", "cum_purchase_pct"]].copy()

    top1_pct  = item_counts[item_counts["item_pct"] <= 1]["cum_purchase_pct"].max()
    top10_pct = item_counts[item_counts["item_pct"] <= 10]["cum_purchase_pct"].max()

    weekly_pop = (
        transactions.withColumn("week", F.date_trunc("week", F.col(date_col)))
          .groupBy("week", item_col)
          .agg(F.count("*").alias("n_purchases"))
          .toPandas()
    )
    weekly_pop["week"] = weekly_pop["week"].astype(str)

    weeks = sorted(weekly_pop["week"].unique())
    correlations = []

    for prev_week, curr_week in zip(weeks[:-1], weeks[1:]):
        prev_week_pop = (
            weekly_pop[weekly_pop["week"] == prev_week]
            .nlargest(top_n, "n_purchases")
            .set_index(item_col)["n_purchases"]
        )
        curr_week_pop = (
            weekly_pop[weekly_pop["week"] == curr_week]
            .set_index(item_col)["n_purchases"]
        )
        shared_items = prev_week_pop.index.intersection(curr_week_pop.index)
        if len(shared_items) > 10:
            corr, _ = spearmanr(prev_week_pop[shared_items], curr_week_pop[shared_items])
            correlations.append({"week": curr_week, "spearman_r": corr})

    weekly_ranks = pd.DataFrame(correlations)
    mean_corr    = weekly_ranks["spearman_r"].mean()

    summary = pd.DataFrame([{
        "top1_pct"       : top1_pct,
        "top10_pct"      : top10_pct,
        "mean_rank_corr" : mean_corr,
    }])

    return {
        "lorenz"      : lorenz,
        "weekly_ranks": weekly_ranks,
        "summary"     : summary,
    }


def plot_demand_concentration(data: dict) -> plt.Figure:
    """
    Plots demand concentration metrics.

    Panel 1: Lorenz curve
    Panel 2: Week-over-week rank correlation
    """
    lorenz       = data["lorenz"]
    weekly_ranks = data["weekly_ranks"]
    summary      = data["summary"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    chart_title(fig, "Demand concentration")

    axes[0].plot(lorenz["item_pct"], lorenz["cum_purchase_pct"],
                 color="steelblue", linewidth=2, label="Actual")
    axes[0].plot([0, 100], [0, 100], "--", color="gray",
                 alpha=0.5, label="Perfect equality")
    axes[0].axvline(1,  color="firebrick",  linestyle=":",
                    alpha=0.8, label=f"Top 1% → {summary['top1_pct']:.1f}% of purchases")
    axes[0].axvline(10, color="darkorange", linestyle=":",
                    alpha=0.8, label=f"Top 10% → {summary['top10_pct']:.1f}% of purchases")
    axes[0].set_xlabel("% of items (ranked by popularity)")
    axes[0].set_ylabel("% of total purchases")
    axes[0].set_title("Lorenz Curve")
    axes[0].legend(fontsize=9)
    axes[0].grid(True, alpha=0.25)

    axes[1].plot(weekly_ranks["week"], weekly_ranks["spearman_r"],
                 color="darkorange", linewidth=1.5)
    axes[1].fill_between(weekly_ranks["week"], weekly_ranks["spearman_r"],
                         alpha=0.15, color="darkorange")
    axes[1].axhline(summary["mean_rank_corr"], color="firebrick", linestyle="--",
                    label=f"Mean: {summary['mean_rank_corr']:.2f}")
    axes[1].set_xlabel("week")
    axes[1].set_ylabel("Spearman rank correlation")
    axes[1].set_title("Week-over-Week Popularity Stability")
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(8))
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()

    print(f"\n  Concentration & stability:")
    print(f"     Top 1% of items account for {summary['top1_pct']:.1f}% of purchases.")
    print(f"     Top 10% of items account for {summary['top10_pct']:.1f}% of purchases.")
    print(f"     Mean week-over-week rank correlation: {summary['mean_rank_corr']:.2f}")

    return fig


def compute_repurchase(
    transactions: DataFrame,
    date_col: str = "t_dat",
    user_col: str = "customer_id",
    item_col: str = "article_id",
) -> dict:
    """
    Computes repurchase behaviour metrics.

    Returns
    -------
    dict with keys:
        "repurchase_rate"   : float — % of users who bought the same item >1x
        "repurchase_share"  : float — % of all transactions that are repurchases
        "time_to_repurchase": pd.DataFrame — distribution of days between 1st and 2nd purchase of same item
        "summary"           : pd.DataFrame — one-row summary of all three metrics
    """
    user_item_window = Window.partitionBy(user_col, item_col).orderBy(date_col)
    ranked_purchases = transactions.withColumn("purchase_rank", F.rank().over(user_item_window))

    total_users      = transactions.select(user_col).distinct().count()
    repurchase_users = (
        ranked_purchases.filter(F.col("purchase_rank") == 2)
              .select(user_col).distinct().count()
    )
    repurchase_rate = repurchase_users / total_users * 100

    first_purchase  = ranked_purchases.filter(F.col("purchase_rank") == 1).select(
        user_col, item_col, F.col(date_col).alias("first_date")
    )
    second_purchase = ranked_purchases.filter(F.col("purchase_rank") == 2).select(
        user_col, item_col, F.col(date_col).alias("second_date")
    )

    time_to_repurchase = (
        first_purchase.join(second_purchase, on=[user_col, item_col], how="inner")
        .withColumn("days_to_repurchase", F.datediff("second_date", "first_date"))
        .groupBy("days_to_repurchase")
        .agg(F.count("*").alias("n_pairs"))
        .orderBy("days_to_repurchase")
        .toPandas()
    )

    total_transactions      = transactions.count()
    repurchase_transactions = ranked_purchases.filter(F.col("purchase_rank") >= 2).count()
    repurchase_share        = repurchase_transactions / total_transactions * 100

    # weight by n_pairs (time_to_repurchase is a frequency table, one row per distinct gap size)
    total_repurchase_pairs = time_to_repurchase["n_pairs"].sum()
    mean_days = (time_to_repurchase["days_to_repurchase"] * time_to_repurchase["n_pairs"]).sum() / total_repurchase_pairs
    ordered_pairs = time_to_repurchase.sort_values("days_to_repurchase")
    median_days = ordered_pairs.loc[
        ordered_pairs["n_pairs"].cumsum().values >= total_repurchase_pairs / 2,
        "days_to_repurchase"].iloc[0]
    summary = pd.DataFrame([{
        "repurchase_rate_pct"      : round(repurchase_rate, 2),
        "repurchase_share_pct"     : round(repurchase_share, 2),
        "median_days_to_repurchase": median_days,
        "mean_days_to_repurchase"  : round(mean_days, 1),
    }])

    return {
        "repurchase_rate"   : repurchase_rate,
        "repurchase_share"  : repurchase_share,
        "time_to_repurchase": time_to_repurchase,
        "summary"           : summary,
    }


def plot_repurchase(data: dict) -> plt.Figure:
    """
    Panel 1: Summary metrics as text/bar
    Panel 2: Cumulative % of repurchases by days since first purchase
    """
    ttr     = data["time_to_repurchase"]
    summary = data["summary"].iloc[0]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    chart_title(fig, "Repurchase behaviour")

    values  = [summary["repurchase_rate_pct"], summary["repurchase_share_pct"]]
    labels  = ["% users who\nrepurchased same item", "% transactions\nthat are repurchases"]
    bars = axes[0].bar(labels, values, color=["steelblue", "darkorange"], alpha=0.85)
    for bar, val in zip(bars, values):
        axes[0].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                     f"{val:.1f}%", ha="center", va="bottom", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("%")
    axes[0].set_title("Repurchase Rate & Volume Share")
    axes[0].set_ylim(0, max(values) * 1.3)
    axes[0].grid(True, alpha=0.25, axis="y")

    ttr_sorted = ttr.sort_values("days_to_repurchase")
    cum_pct    = ttr_sorted["n_pairs"].cumsum() / ttr_sorted["n_pairs"].sum() * 100
    axes[1].plot(ttr_sorted["days_to_repurchase"], cum_pct, color="darkorange", linewidth=2)
    axes[1].fill_between(ttr_sorted["days_to_repurchase"], cum_pct, alpha=0.15, color="darkorange")
    for pct, color in [(50, "firebrick"), (80, "seagreen")]:
        row = ttr_sorted[cum_pct >= pct]
        if not row.empty:
            day = row.iloc[0]["days_to_repurchase"]
            axes[1].axhline(pct, color=color, linestyle="--", alpha=0.7, linewidth=1,
                            label=f"{pct}% repurchase within {day:.0f} days")
            axes[1].axvline(day, color=color, linestyle="--", alpha=0.7, linewidth=1)
    axes[1].set_xlabel("days since first purchase")
    axes[1].set_ylabel("cumulative % of repurchases")
    axes[1].set_title("Cumulative Repurchase Timing")
    axes[1].set_xlim(0, 365)
    axes[1].legend(fontsize=9)
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()

    print(f"\n  Repurchase behaviour:")
    print(f"     {summary['repurchase_rate_pct']:.1f}% of users ever repurchased the same item.")
    print(f"     {summary['repurchase_share_pct']:.1f}% of all transactions are repurchases.")
    print(f"     Median time to repurchase: {summary['median_days_to_repurchase']:.0f} days.")

    return fig


def compute_user_activity_segments(
    df: DataFrame,
    date_col: str = "t_dat",
    user_col: str = "customer_id",
    reference_date: str = "2020-09-20",
) -> dict:
    """
    Computes user activity segments based on recency of last purchase.

    Parameters
    ----------
    df             : transactions Spark DataFrame
    date_col       : date column name
    user_col       : user column name
    reference_date : date to measure recency from (default: last full week)

    Returns
    -------
    dict with keys:
        "days_since_last"  : pd.DataFrame — distribution of days since last purchase
        "segments"         : pd.DataFrame — user counts per activity segment
    """

    user_last_purchase = (
        df.groupBy(user_col)
          .agg(F.max(date_col).alias("last_purchase_date"))
          .withColumn(
              "days_since_last",
              F.datediff(F.lit(reference_date), F.col("last_purchase_date"))
          )
    )

    user_segments = (
        user_last_purchase
        .withColumn(
            "segment",
            F.when(F.col("days_since_last") <= 28,  "1_active")
             .when(F.col("days_since_last") <= 84,  "2_lapsed")
             .when(F.col("days_since_last") <= 365, "3_inactive")
             .otherwise("4_gone")
        )
    )

    segments = (
        user_segments
        .groupBy("segment")
        .agg(F.count("*").alias("n_users"))
        .orderBy("segment")
        .withColumn(
            "pct_users",
            F.col("n_users") / F.lit(user_segments.count()) * 100
        )
        .toPandas()
    )

    # Bucket into 2-week bins to avoid collecting one row per user
    days_dist = (
        user_last_purchase
        .withColumn("bucket", F.floor(F.col("days_since_last") / 14) * 14)
        .groupBy("bucket")
        .agg(F.count("*").alias("n_users"))
        .orderBy("bucket")
        .toPandas()
    )

    return {
        "days_since_last": days_dist,
        "segments"       : segments,
    }


def plot_user_activity_segments(data: dict) -> plt.Figure:
    """
    Plots user activity segments.

    Panel 1: Days since last purchase distribution
    Panel 2: User segment breakdown
    """
    days_dist = data["days_since_last"]
    segments  = data["segments"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    chart_title(fig, "Active vs inactive users")

    axes[0].bar(
        days_dist["bucket"],
        days_dist["n_users"],
        color="steelblue", edgecolor="white", alpha=0.85, width=12,
    )
    for day, label, color in [
        (28,  "active/lapsed",   "firebrick"),
        (84,  "lapsed/inactive", "darkorange"),
        (365, "inactive/gone",   "seagreen"),
    ]:
        axes[0].axvline(day, color=color, linestyle="--",
                        alpha=0.8, linewidth=1.5, label=label)
    axes[0].set_xlabel("days since last purchase")
    axes[0].set_ylabel("# users")
    axes[0].set_title("Days Since Last Purchase")
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.25)

    segment_labels = {
        "1_active"  : "Active\n(≤4 weeks)",
        "2_lapsed"  : "Lapsed\n(4-12 weeks)",
        "3_inactive": "Inactive\n(12-52 weeks)",
        "4_gone"    : "Gone\n(>1 year)",
    }
    colors = ["seagreen", "steelblue", "darkorange", "firebrick"]

    bars = axes[1].bar(
        [segment_labels[s] for s in segments["segment"]],
        segments["pct_users"],
        color=colors, edgecolor="white", alpha=0.85,
    )
    axes[1].set_xlabel("segment")
    axes[1].set_ylabel("% of users")
    axes[1].set_title("User Activity Segments")
    for bar, val in zip(bars, segments["pct_users"]):
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontsize=9
        )
    axes[1].grid(True, alpha=0.25)

    plt.tight_layout()

    active_pct   = segments[segments["segment"] == "1_active"]["pct_users"].values[0]
    lapsed_pct   = segments[segments["segment"] == "2_lapsed"]["pct_users"].values[0]
    inactive_pct = segments[segments["segment"] == "3_inactive"]["pct_users"].values[0]
    gone_pct     = segments[segments["segment"] == "4_gone"]["pct_users"].values[0]

    print(f"\n  Activity segments:")
    print(f"     Active   (≤4 weeks)   : {active_pct:.1f}% of users")
    print(f"     Lapsed   (4-12 weeks) : {lapsed_pct:.1f}% of users")
    print(f"     Inactive (12-52 weeks): {inactive_pct:.1f}% of users")
    print(f"     Gone     (>1 year)    : {gone_pct:.1f}% of users")

    return fig


def make_fold(transactions, week_end, train_weeks=None):
    """history = everything before this week; next_purchases = the (customer, item) bought this week.

    Boundaries come from `recolib.data.TimeSeriesSplitter` — the same splitter the model
    pipeline uses — so the analysis here and the pipeline cannot drift apart. The one
    difference is deliberate: `train_weeks=None` gives unbounded history, because this
    notebook asks whether a signal exists at all, while the pipeline bounds its training
    window. The validation week is identical either way.
    """
    fold = TimeSeriesSplitter(
        window_end=week_end, train_weeks=train_weeks,
        val_weeks=1, n_folds=1, with_submission=False,
    ).split(transactions)[0]
    next_purchases = fold.val.select(CUSTOMER, ITEM).distinct()
    return fold.train, next_purchases


def evaluate_source(name, candidates, next_purchases, n_next, n_buyers, n_customers, k_global=None):
    if k_global is None:                                  # personalized source
        candidate_pairs = candidates.select(CUSTOMER, ITEM).distinct()
        n_candidates = candidate_pairs.count(); hits = candidate_pairs.join(next_purchases, [CUSTOMER, ITEM])
    else:                                                 # same items for everyone
        n_candidates = n_customers * k_global; hits = next_purchases.join(F.broadcast(candidates), ITEM)
    n_hits = hits.count(); n_buyers_hit = hits.select(CUSTOMER).distinct().count()
    return {"source": name, "recall %": 100*n_hits/n_next,
            "precision %": 100*n_hits/n_candidates if n_candidates else 0.0,
            "hit-rate %": 100*n_buyers_hit/n_buyers}


def consistency(usual_df, next_df):
    joined = usual_df.join(next_df, CUSTOMER).cache()
    stats = joined.agg(F.count("*").alias("n"),
                   F.avg((F.col("usual") == F.col("next_val")).cast("int")).alias("obs")).collect()[0]
    n, observed = stats["n"], stats["obs"]
    p_usual = {fold[0]: fold[1] / n for fold in joined.groupBy("usual").count().collect()}
    p_next  = {fold[0]: fold[1] / n for fold in joined.groupBy("next_val").count().collect()}
    joined.unpersist()
    return observed, sum(p_usual.get(value, 0) * p_next.get(value, 0) for value in set(p_usual) | set(p_next))


def price_tier(col, cuts):
    return F.when(col <= cuts[0], "budget").when(col <= cuts[1], "mid").otherwise("premium")


def top_index(rows, alias, index_group):
    """Each customer's most-bought index_group value, returned as column `alias`."""
    return (rows.join(index_group, ITEM).groupBy(CUSTOMER, "index_group_no").count()
            .withColumn("rk", F.row_number().over(Window.partitionBy(CUSTOMER).orderBy(F.col("count").desc())))
            .filter(F.col("rk") == 1).select(CUSTOMER, F.col("index_group_no").alias(alias)))
