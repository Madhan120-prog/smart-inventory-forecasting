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
