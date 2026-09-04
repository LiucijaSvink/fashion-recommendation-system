"""Single source of truth for the column names used across recolib.

Library code never hard-codes column names — it reads them from a `Schema` instance
(passed via `RetrievalContext`, `LambdaRanker(schema=...)`, etc.). The defaults match
H&M; override by constructing your own `Schema`:

    from recolib import Schema
    schema = Schema(user="my_user_id", item="my_product_id", date="ts")

If you keep your column names in a JSON / YAML / TOML file, load it yourself and
pass the values in — the library doesn't care where they come from."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Schema:
    """Column-name configuration. Immutable so it can be safely shared across objects."""
    user: str = "customer_id"
    item: str = "article_id"
    date: str = "t_dat"

    @classmethod
    def from_json(cls, path: str | Path) -> "Schema":
        """Tiny convenience: read a JSON file with keys {user, item, date}."""
        with open(path) as f:
            return cls(**json.load(f))


DEFAULT_SCHEMA = Schema()
