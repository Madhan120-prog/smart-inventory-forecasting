"""Forecasting helpers extracted from notebooks/02_store_item_demand_forecast.ipynb."""
from __future__ import annotations

import pandas as pd


def fit_forecast(df: pd.DataFrame, periods: int = 90) -> pd.DataFrame:
    """Fit a Prophet model on one series and forecast ``periods`` days ahead.

    Args:
        df: DataFrame with a ``date`` column and a ``sales`` column.
        periods: Number of future days to predict.

    Returns:
        Prophet's forecast frame (``ds``, ``yhat``, ``yhat_lower``, ...) covering
        the history plus ``periods`` future days.
    """
    from prophet import Prophet  # imported lazily so helpers work without it

    series = df.rename(columns={"date": "ds", "sales": "y"})[["ds", "y"]]
    model = Prophet()
    model.fit(series)
    future = model.make_future_dataframe(periods=periods)
    return model.predict(future)


def summarize_change(history: pd.DataFrame, forecast: pd.DataFrame, window: int = 90) -> dict[str, float]:
    """Compare the last ``window`` days of history with the forecast tail."""
    past_mean = float(history.tail(window)["sales"].mean())
    future_mean = float(forecast.tail(window)["yhat"].mean())
    return {
        "past_sales": round(past_mean, 2),
        "forecasted_sales": round(future_mean, 2),
        "change": round(future_mean - past_mean, 2),
    }


def top_n_items(summary: pd.DataFrame, n: int = 5, by: str = "forecasted_sales") -> pd.DataFrame:
    """Return the ``n`` rows of ``summary`` with the largest ``by`` value, descending."""
    return summary.sort_values(by=by, ascending=False).head(n).reset_index(drop=True)
