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


def mape(actual: pd.Series, predicted: pd.Series) -> float:
    """Mean absolute percentage error (%), ignoring points where ``actual`` is zero."""
    mask = actual != 0
    return float(((actual[mask] - predicted[mask]).abs() / actual[mask].abs()).mean() * 100)


def holdout_split(df: pd.DataFrame, holdout: int = 90) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a dated series into (train, test), reserving the last ``holdout`` rows for testing."""
    ordered = df.sort_values("date").reset_index(drop=True)
    return ordered.iloc[:-holdout], ordered.iloc[-holdout:]


def backtest_mape(df: pd.DataFrame, holdout: int = 90) -> float:
    """Fit on all but the last ``holdout`` days and return the MAPE (%) on those days."""
    train, test = holdout_split(df, holdout)
    forecast = fit_forecast(train, periods=holdout)
    predicted = forecast.tail(holdout)["yhat"].reset_index(drop=True)
    return mape(test["sales"].reset_index(drop=True), predicted)
