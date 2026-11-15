import pandas as pd
import pytest

from forecast import mape, summarize_change, top_n_items


def test_top_n_items_orders_descending_and_limits():
    summary = pd.DataFrame({"item": [1, 2, 3, 4], "forecasted_sales": [10.0, 40.0, 30.0, 20.0]})
    top = top_n_items(summary, n=2)
    assert list(top["item"]) == [2, 3]


def test_summarize_change():
    history = pd.DataFrame({"sales": [10.0] * 5})
    forecast = pd.DataFrame({"yhat": [15.0] * 5})
    assert summarize_change(history, forecast, window=5) == {
        "past_sales": 10.0,
        "forecasted_sales": 15.0,
        "change": 5.0,
    }


def test_fit_forecast_length_and_no_nans():
    pytest.importorskip("prophet")
    from forecast import fit_forecast

    dates = pd.date_range("2020-01-01", periods=60)
    df = pd.DataFrame({"date": dates, "sales": [10 + (i % 7) for i in range(60)]})
    out = fit_forecast(df, periods=10)
    assert len(out) == 70
    assert not out["yhat"].isna().any()


def test_mape_ignores_zero_actuals():
    actual = pd.Series([100.0, 0.0, 50.0])
    predicted = pd.Series([110.0, 5.0, 45.0])
    assert mape(actual, predicted) == pytest.approx(10.0)


def test_holdout_split_keeps_last_rows_for_test():
    from forecast import holdout_split

    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=10)[::-1], "sales": range(10)})
    train, test = holdout_split(df, holdout=3)
    assert len(train) == 7 and len(test) == 3
    assert train["date"].max() < test["date"].min()


def test_seasonal_naive_repeats_last_season():
    from forecast import seasonal_naive_forecast

    history = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = seasonal_naive_forecast(history, periods=5, season=3)
    assert list(out) == [3.0, 4.0, 5.0, 3.0, 4.0]


def test_seasonal_naive_rejects_short_history():
    from forecast import seasonal_naive_forecast

    with pytest.raises(ValueError):
        seasonal_naive_forecast(pd.Series([1.0, 2.0]), periods=3, season=7)


def test_backtest_baseline_mape_is_zero_on_perfectly_weekly_series():
    from forecast import backtest_baseline_mape

    dates = pd.date_range("2020-01-01", periods=42)
    df = pd.DataFrame({"date": dates, "sales": [10.0 + (i % 7) for i in range(42)]})
    assert backtest_baseline_mape(df, holdout=14, season=7) == pytest.approx(0.0)
