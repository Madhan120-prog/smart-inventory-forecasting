"""Render forecast charts for the README into outputs/.

Usage (from the repo root):  python scripts/plot_forecasts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecast import fit_forecast, plot_forecast, select_series  # noqa: E402

SERIES = [(1, 1), (1, 15)]  # (store, item): the same pairs used in the baseline comparison


def main() -> None:
    data = pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["date"])
    for store, item in SERIES:
        history = select_series(data, store, item)
        forecast = fit_forecast(history, periods=90)
        out = plot_forecast(
            history.tail(365),
            forecast[forecast["ds"] >= history["date"].iloc[-365]],
            ROOT / "outputs" / f"store{store}_item{item}_forecast.png",
            title=f"Store {store}, item {item}: last year + 90-day Prophet forecast",
        )
        print(f"wrote {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
