# Smart Inventory Forecasting: Task List

_Last updated by the weekly review routine: 2026-11-15._

This file replaces the separate weekly-plan branches (`weekly-plan-2026-10-20`,
`weekly/forecasting-next-steps`, `weekly/next-steps`, `claude/weekly-python-review`,
`claude/weekly-roadmap`). Each of those added only a planning note, and none were merged,
so they can be deleted. Track everything here on `feat/forecast-module`.

## Done (on this branch)
- [x] `forecast.py` module pulled out of the notebook: `fit_forecast`, `summarize_change`, `top_n_items` (with type hints)
- [x] `mape`, `holdout_split`, `backtest_mape` evaluation helpers
- [x] `seasonal_naive_forecast` + `backtest_baseline_mape` baseline (2026-11-15)
- [x] pytest suite: 7 passing, 1 skipped when Prophet isn't installed
- [x] `requirements.txt`, plus a how-to-run section in the README

## Next step (pick this up first)
- [ ] **Run the Prophet-vs-baseline comparison on the real data.** Install Prophet, then for 3–5
      store/item series in `data/train.csv` compute `backtest_mape` and `backtest_baseline_mape`.
      Save the results to `outputs/baseline_comparison.csv` and put a short table in the README.
      If Prophet doesn't beat the seasonal-naive baseline, that's worth stating too.
- [ ] **Open a PR from `feat/forecast-module` into `main` and merge it.** `main` hasn't changed since 2025-11-10.

## Housekeeping
- [ ] Make the notebook import from `forecast.py` instead of re-defining the logic
- [ ] Fix the README folder structure: notebook names don't match (`Automobile Demand Forecasting .ipynb` has a trailing space)
- [ ] Replace the placeholder `user-images.githubusercontent.com/example/...` image links with real charts from `outputs/`
- [ ] Decide whether the 17 MB `data/train.csv` stays in git (or `.gitignore` it and add a download note)
- [ ] Delete the five stale weekly-plan branches listed above

## Later
- [ ] GitHub Actions workflow that runs `pytest` on every push
- [ ] Streamlit dashboard for top-N forecasts
