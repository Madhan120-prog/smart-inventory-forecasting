"""Backtest Prophet against the seasonal-naive baseline on every store/item series.

Usage:
    python scripts/backtest_all.py [--holdout 90] [--limit N]

Writes outputs/backtest_mape.csv (one row per series) and prints a win count.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecast import backtest_all  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=int, default=90)
    parser.add_argument("--limit", type=int, default=None, help="only the first N store-item pairs")
    args = parser.parse_args()

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)

    data = pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["date"])
    result = backtest_all(data, holdout=args.holdout, limit=args.limit)

    out = ROOT / "outputs" / "backtest_mape.csv"
    result.to_csv(out, index=False)
    wins = int((result["winner"] == "prophet").sum())
    print(f"wrote {out.relative_to(ROOT)}")
    print(f"Prophet beats seasonal naive on {wins} of {len(result)} series")
    print(result[["prophet_mape", "seasonal_naive_mape"]].describe().round(2).to_string())


if __name__ == "__main__":
    main()
