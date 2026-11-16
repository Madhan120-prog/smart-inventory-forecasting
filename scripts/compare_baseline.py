"""Compare Prophet against the seasonal-naive baseline on a few store/item series.

Usage:
    python scripts/compare_baseline.py [--holdout 90]

Writes outputs/baseline_comparison.csv.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecast import backtest_baseline_mape, backtest_mape  # noqa: E402

SERIES: list[tuple[int, int]] = [(1, 1), (1, 15), (2, 28), (5, 10), (10, 45)]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--holdout", type=int, default=90)
    args = parser.parse_args()

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)

    data = pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["date"])
    rows = []
    for store, item in SERIES:
        series = data[(data["store"] == store) & (data["item"] == item)][["date", "sales"]]
        prophet = backtest_mape(series, holdout=args.holdout)
        baseline = backtest_baseline_mape(series, holdout=args.holdout)
        rows.append(
            {
                "store": store,
                "item": item,
                "prophet_mape": round(prophet, 2),
                "seasonal_naive_mape": round(baseline, 2),
                "winner": "prophet" if prophet < baseline else "seasonal_naive",
            }
        )
        print(rows[-1])

    result = pd.DataFrame(rows)
    out = ROOT / "outputs" / "baseline_comparison.csv"
    result.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
