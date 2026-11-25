"""Render forecast charts for the README into outputs/.

Usage (from the repo root):  python scripts/plot_forecasts.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecast import fit_forecast, plot_components, plot_forecast, select_series  # noqa: E402

SERIES = [(1, 1), (1, 15)]  # (store, item): the same pairs used in the baseline comparison
COMPONENTS_SERIES = (1, 1)  # pair whose trend/weekly/yearly components go in the README


def plot_store_items(data: pd.DataFrame) -> None:
    for store, item in SERIES:
        history = select_series(data, store, item)
        model, forecast = fit_forecast(history, periods=90, return_model=True)
        out = plot_forecast(
            history.tail(365),
            forecast[forecast["ds"] >= history["date"].iloc[-365]],
            ROOT / "outputs" / f"store{store}_item{item}_forecast.png",
            title=f"Store {store}, item {item}: last year + 90-day Prophet forecast",
        )
        print(f"wrote {out.relative_to(ROOT)}")
        if (store, item) == COMPONENTS_SERIES:
            out = plot_components(model, forecast, ROOT / "outputs" / f"store{store}_item{item}_components.png")
            print(f"wrote {out.relative_to(ROOT)}")


def plot_brake_pads() -> None:
    """Chart the automobile notebook's saved Brake Pads forecast against its daily history."""
    sales = pd.read_csv(ROOT / "data" / "sales_data.csv", parse_dates=["Date"])
    daily = sales[sales["Product"] == "Brake Pads"].groupby("Date")["Units_Sold"].sum().reset_index()
    history = daily.rename(columns={"Date": "date", "Units_Sold": "sales"})
    forecast = pd.read_csv(ROOT / "outputs" / "Brake Pads_forecast.csv", parse_dates=["ds"])
    out = plot_forecast(
        history,
        forecast,
        ROOT / "outputs" / "brake_pads_forecast.png",
        title="Brake Pads: daily units sold + 30-day Prophet forecast",
    )
    print(f"wrote {out.relative_to(ROOT)}")


def main() -> None:
    data = pd.read_csv(ROOT / "data" / "train.csv", parse_dates=["date"])
    plot_store_items(data)
    plot_brake_pads()


if __name__ == "__main__":
    main()
