"""Spark session helper."""
from __future__ import annotations

import os
import sys

from pyspark.sql import SparkSession


def get_spark(
    app_name: str = "recolib",
    driver_memory: str = "12g",
    shuffle_partitions: int = 400,
    console_progress: bool = True,
) -> SparkSession:
    """Build (or fetch) a SparkSession tuned for single-node candidate/feature work.

    `console_progress=False` silences the stage progress bars. They are useful in a
    terminal and unreadable in a saved notebook, where they bury the actual results —
    and the setting is read when the context is created, so it cannot be changed later.
    """
    # Python workers default to whatever `python3` resolves to, which is not the
    # interpreter this process runs in — so a UDF or mapInPandas cannot import the
    # library. Pinning both ends to sys.executable is what makes executor-side
    # prediction work at all.
    os.environ.setdefault("PYSPARK_PYTHON", sys.executable)
    os.environ.setdefault("PYSPARK_DRIVER_PYTHON", sys.executable)

    spark = (
        SparkSession.builder.appName(app_name)
        .config("spark.driver.memory", driver_memory)
        .config("spark.driver.maxResultSize", "4g")
        .config("spark.sql.shuffle.partitions", str(shuffle_partitions))
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.sql.adaptive.skewJoin.enabled", "true")
        .config("spark.ui.showConsoleProgress", "true" if console_progress else "false")
        .getOrCreate()
    )
    return spark
