"""Forecasting helpers extracted from notebooks/02_store_item_demand_forecast.ipynb."""
from __future__ import annotations

from collections.abc import Callable, Iterable

import pandas as pd

Forecaster = Callable[[pd.DataFrame, int], pd.DataFrame]


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


def select_series(data: pd.DataFrame, store: int, item: int) -> pd.DataFrame:
    """Return the ``date``/``sales`` rows for one store-item pair, sorted by date."""
    mask = (data["store"] == store) & (data["item"] == item)
    if not mask.any():
        raise ValueError(f"no rows for store={store}, item={item}")
    return data.loc[mask, ["date", "sales"]].sort_values("date").reset_index(drop=True)


def build_forecast_summary(
    data: pd.DataFrame,
    store: int,
    items: Iterable[int],
    periods: int = 90,
    forecaster: Forecaster | None = None,
) -> pd.DataFrame:
    """Forecast each item in ``store`` and tabulate past vs forecasted mean sales.

    This is the per-item loop from the notebook. ``forecaster`` defaults to
    :func:`fit_forecast` (Prophet); tests pass a cheap stand-in instead.

    Returns:
        Columns ``Item ID``, ``Past Sales``, ``Forecasted Sales``, ``Change``,
        sorted by ``Change`` descending (the layout of
        ``outputs/store1_top5_forecast_summary.csv``).
    """
    fit = forecaster or fit_forecast
    rows = []
    for item in items:
        history = select_series(data, store, item)
        stats = summarize_change(history, fit(history, periods), window=periods)
        rows.append(
            {
                "Item ID": item,
                "Past Sales": stats["past_sales"],
                "Forecasted Sales": stats["forecasted_sales"],
                "Change": stats["change"],
            }
        )
    summary = pd.DataFrame(rows, columns=["Item ID", "Past Sales", "Forecasted Sales", "Change"])
    return summary.sort_values("Change", ascending=False).reset_index(drop=True)


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


def seasonal_naive_forecast(history: pd.Series, periods: int, season: int = 7) -> pd.Series:
    """Baseline forecast: repeat the last ``season`` observed values for ``periods`` steps.

    With ``season=7`` on daily data this predicts "same as the same weekday last week",
    which is the standard baseline Prophet should beat to be worth its complexity.
    """
    if len(history) < season:
        raise ValueError(f"need at least {season} observations, got {len(history)}")
    last_season = history.iloc[-season:].to_numpy()
    values = [last_season[i % season] for i in range(periods)]
    return pd.Series(values, name="yhat", dtype=float)


def backtest_baseline_mape(df: pd.DataFrame, holdout: int = 90, season: int = 7) -> float:
    """Seasonal-naive counterpart of ``backtest_mape`` (no Prophet needed)."""
    train, test = holdout_split(df, holdout)
    predicted = seasonal_naive_forecast(train["sales"], periods=holdout, season=season)
    return mape(test["sales"].reset_index(drop=True), predicted)
