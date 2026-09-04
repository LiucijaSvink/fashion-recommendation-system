"""Command-line entry point.

    recolib run --config configs/config.json                  # everything
    recolib run --config configs/config.json --stages train evaluate
    recolib run --config configs/config.json --folds fold_0 fold_1 --max-candidates 200
    recolib show --config configs/config.json                 # print the resolved config

Any config field can be overridden with `--set key=value`, so one JSON file plus flags
covers experiments without editing code.
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from typing import Any

from .backends import get_spark
from .config import PipelineConfig
from .pipeline import STAGES, run_all


def _coerce(value: str) -> Any:
    """Turn a CLI string into the JSON value it looks like ('12' -> 12, 'null' -> None)."""
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _build_config(args: argparse.Namespace) -> PipelineConfig:
    cfg = PipelineConfig.from_json(args.config)
    overrides: dict[str, Any] = {}
    if args.folds:
        overrides["train_folds"] = tuple(args.folds)
    if args.max_candidates is not None:
        overrides["max_candidates"] = args.max_candidates
    if args.eval_users is not None:
        overrides["eval_sample_users"] = args.eval_users or None
    if args.seed is not None:
        overrides["seed"] = args.seed
    for item in args.set or []:
        if "=" not in item:
            raise SystemExit(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        overrides[key] = _coerce(raw)
    return cfg.with_(**overrides) if overrides else cfg


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", required=True, help="path to a pipeline config JSON")
    parser.add_argument("--folds", nargs="+", metavar="FOLD",
                        help="folds to train on, e.g. fold_0 fold_1")
    parser.add_argument("--max-candidates", type=int, help="per-customer retrieval cap")
    parser.add_argument("--eval-users", type=int,
                        help="customers to evaluate on (0 = all, scored in chunks)")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--set", action="append", metavar="KEY=VALUE",
                        help="override any config field; repeatable")


def build_parser() -> argparse.ArgumentParser:
    """The `recolib run` / `recolib show` command line."""
    parser = argparse.ArgumentParser(prog="recolib", description=__doc__.split("\n")[0])
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run pipeline stages")
    _add_common(run)
    run.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES),
                     help=f"stages to run (default: all of {', '.join(STAGES)})")
    run.add_argument("--overwrite", action="store_true",
                     help="rewrite fold parquets even if they already exist")

    show = sub.add_parser("show", help="print the resolved configuration and exit")
    _add_common(show)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point: resolve the config, run the requested stages, return an exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s  %(levelname)-7s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    try:
        cfg = _build_config(args)
    except (ValueError, FileNotFoundError) as error:
        raise SystemExit(f"recolib: {error}") from None

    if args.command == "show":
        print(cfg.summary())
        return 0

    print(cfg.summary(), file=sys.stderr)
    spark = get_spark("recolib-pipeline", driver_memory=cfg.driver_memory,
                      shuffle_partitions=cfg.shuffle_partitions)
    spark.sparkContext.setLogLevel("WARN")
    try:
        run_all(spark, cfg, stages=tuple(args.stages), overwrite=args.overwrite)
    finally:
        spark.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
