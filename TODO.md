# Weekly Focus: Smart Inventory Forecasting (week of 2026-11-05)

**Status:** Stalled. Last commit was 2025-11-10 (README edits). The code exists only in two notebooks, and there are no `.py` modules or tests.

## This week's 2 improvements
- [ ] **1. Extract notebook logic into a `src/forecasting.py` module with type hints.** Functions: `load_sales(path) -> pd.DataFrame`, `fit_prophet(df, horizon) -> pd.DataFrame`, `top_n_items(forecast_df, n) -> pd.DataFrame`. Notebooks then import from `src/`.
- [ ] **2. Add `tests/test_forecasting.py` (pytest).** Cover `top_n_items` and the loader on a tiny synthetic DataFrame. No Prophet fit is needed for the unit tests. Add `requirements.txt`.

## Housekeeping (small wins)
- [ ] Fix README folder structure: the notebook names don't match the actual files (`Automobile Demand Forecasting .ipynb` has a trailing space; the README says `01_Automobile_Demand_Forecasting.ipynb`).
- [ ] Replace the placeholder `user-images.githubusercontent.com/example/...` image links with real charts saved in `outputs/`.
- [ ] Decide whether to keep the 17 MB `data/train.csv` in git (consider `.gitignore` plus a download note).

## Later
- Baseline comparison (naive / ARIMA) with MAPE metric vs Prophet.
- Streamlit dashboard.
