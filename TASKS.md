# Weekly Plan — week of 2026-10-20

**Status:** notebook-only project, no `.py` modules, no tests. Last substantive commit was 2025-11-10 (stalled ~11 months).

## This week's focus: turn notebook logic into a tested Python module
- [ ] Create `src/forecast.py` with `forecast_product(df, product, horizon=30)` (Prophet fit + forecast) and `stock_risk(forecast_df, window=30)` (% change recent vs forecast), with type hints and docstrings.
- [ ] Add `tests/test_forecast.py` (pytest) using a tiny synthetic series: output length == horizon, no NaNs, `stock_risk` sign is correct.

## Next up
- [ ] Fix README: image links point to `user-images.githubusercontent.com/example/...` placeholders; replace with real exported PNGs in `outputs/`.
- [ ] README folder tree lists `01_Automobile_Demand_Forecasting.ipynb`, but the actual file is `Automobile Demand Forecasting .ipynb` — rename the file.
- [ ] Add `requirements.txt` (prophet, pandas, numpy, matplotlib, seaborn, pytest).
- [ ] Compare Prophet vs. a naive/ARIMA baseline using MAPE on a holdout.
