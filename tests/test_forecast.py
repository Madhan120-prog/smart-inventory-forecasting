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


def _store_data() -> pd.DataFrame:
    dates = pd.date_range("2020-01-01", periods=10)
    frames = [
        pd.DataFrame({"date": dates, "store": 1, "item": item, "sales": float(item * 10)})
        for item in (1, 2, 3)
    ]
    frames.append(pd.DataFrame({"date": dates, "store": 2, "item": 1, "sales": 999.0}))
    return pd.concat(frames, ignore_index=True).sample(frac=1, random_state=0)


def test_select_series_filters_and_sorts():
    from forecast import select_series

    out = select_series(_store_data(), store=1, item=2)
    assert list(out.columns) == ["date", "sales"]
    assert len(out) == 10 and (out["sales"] == 20.0).all()
    assert out["date"].is_monotonic_increasing


def test_select_series_rejects_unknown_pair():
    from forecast import select_series

    with pytest.raises(ValueError):
        select_series(_store_data(), store=3, item=1)


def test_build_forecast_summary_with_stub_forecaster():
    from forecast import build_forecast_summary

    def grow_by_item(history: pd.DataFrame, periods: int) -> pd.DataFrame:
        # Pretend each item's demand rises by its own base level / 10.
        base = history["sales"].iloc[0]
        return pd.DataFrame({"yhat": [base + base / 10] * periods})

    summary = build_forecast_summary(_store_data(), store=1, items=[1, 2, 3], periods=5, forecaster=grow_by_item)
    assert list(summary.columns) == ["Item ID", "Past Sales", "Forecasted Sales", "Change"]
    assert list(summary["Item ID"]) == [3, 2, 1]  # sorted by Change, largest first
    assert summary.loc[0].to_dict() == {"Item ID": 3, "Past Sales": 30.0, "Forecasted Sales": 33.0, "Change": 3.0}


def test_build_forecast_summary_with_prophet_matches_committed_output():
    """End-to-end smoke test: the notebook's real Prophet path on store 1, item 1.

    Re-derives one row of outputs/store1_top5_forecast_summary.csv from data/train.csv,
    so a change to forecast.py or the pinned Prophet version that shifts results is caught.
    """
    pytest.importorskip("prophet")
    from pathlib import Path

    from forecast import build_forecast_summary

    root = Path(__file__).resolve().parent.parent
    train = pd.read_csv(root / "data" / "train.csv", parse_dates=["date"])
    expected = pd.read_csv(root / "outputs" / "store1_top5_forecast_summary.csv").set_index("Item ID").loc[1]

    summary = build_forecast_summary(train, store=1, items=[1], periods=90)

    assert list(summary.columns) == ["Item ID", "Past Sales", "Forecasted Sales", "Change"]
    row = summary.loc[0]
    assert row["Item ID"] == 1
    assert row["Past Sales"] == pytest.approx(expected["Past Sales"], abs=0.01)
    assert row["Forecasted Sales"] == pytest.approx(expected["Forecasted Sales"], abs=0.05)
    assert row["Change"] == pytest.approx(expected["Change"], abs=0.05)


def test_plot_forecast_writes_png(tmp_path):
    pytest.importorskip("matplotlib")
    from forecast import plot_forecast

    dates = pd.date_range("2020-01-01", periods=30)
    history = pd.DataFrame({"date": dates[:20], "sales": [10.0 + i % 7 for i in range(20)]})
    forecast = pd.DataFrame(
        {"ds": dates, "yhat": [12.0] * 30, "yhat_lower": [10.0] * 30, "yhat_upper": [14.0] * 30}
    )
    out = plot_forecast(history, forecast, tmp_path / "charts" / "item.png", title="test")
    assert out.exists()
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


def test_plot_components_writes_png(tmp_path):
    pytest.importorskip("prophet")
    from forecast import fit_forecast, plot_components

    dates = pd.date_range("2020-01-01", periods=400)
    history = pd.DataFrame({"date": dates, "sales": [20.0 + (i % 7) + i * 0.01 for i in range(400)]})
    model, forecast = fit_forecast(history, periods=14, return_model=True)
    assert len(forecast) == 414
    out = plot_components(model, forecast, tmp_path / "components.png")
    assert out.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
