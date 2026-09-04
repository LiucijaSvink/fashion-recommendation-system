"""Shared pytest fixtures. The `spark` fixture is session-scoped — one SparkSession
is reused across all tests in the session (Spark startup is the slow bit, ~5–10 s)."""
from __future__ import annotations

import pytest
from pyspark.sql import SparkSession


@pytest.fixture(scope="session")
def spark():
    sess = (
        SparkSession.builder
        .appName("recolib-tests")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "2")
        .config("spark.driver.memory", "1g")
        .config("spark.ui.enabled", "false")
        .config("spark.driver.bindAddress", "127.0.0.1")
        .getOrCreate()
    )
    sess.sparkContext.setLogLevel("ERROR")
    yield sess
    sess.stop()
