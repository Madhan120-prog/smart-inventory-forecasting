# Smart Inventory Forecasting: Task List

_Last updated by the weekly review routine: 2026-11-20._

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
- [x] Prophet-vs-baseline comparison on 5 real series: `scripts/compare_baseline.py`, results in
      `outputs/baseline_comparison.csv` and the README. Prophet wins on all 5 (MAPE 9–23% vs 20–39%). (2026-11-16)
- [x] Full test suite verified with Prophet installed: 8 passed, 0 skipped (2026-11-16)
- [x] GitHub Actions workflow `.github/workflows/tests.yml` runs `pytest` on every push and PR
      (pandas + pytest only; the Prophet test skips). Verified locally: 7 passed, 1 skipped. (2026-11-19)
- [x] `select_series` and `build_forecast_summary` added to `forecast.py` (the notebook's per-item loop), with
      3 new tests that use a stub forecaster so they don't need Prophet. Suite: 11 passed. (2026-11-20)
- [x] `02_store_item_demand_forecast.ipynb` now imports from `forecast.py`, reads `../data/train.csv` (the old
      `store_item_demand_data/` path didn't exist in the repo), writes to `../outputs/`, and was re-run top to bottom.
      `store1_top5_forecast_summary.csv` changed by at most 0.02 on items 2 and 5. The original inline code gives the
      same new numbers under Prophet 1.5.0, so the change comes from the Prophet version, not the refactor. (2026-11-20)
- [x] `ruff check .` (rules E, F, I; config in `ruff.toml`, notebooks excluded) added to the CI workflow. Passes. (2026-11-20)

## Next step (pick this up first)
- [ ] **Open a PR from `feat/forecast-module` into `main` and merge it.** This is the blocker: `main`
      hasn't changed since 2025-11-10, so none of the module, tests or results show on the public repo.
      Takes ~5 minutes: review the diff on GitHub, then merge.
      The PR now gets a green/red CI check, so you can see the tests pass before merging.
- [ ] **Pin library versions in `requirements.txt`** (e.g. `prophet==1.5.0`, plus pandas) so re-running the notebook
      gives the same numbers every time. This week's re-run showed results move slightly between Prophet versions.

## Housekeeping
- [ ] Fix the README folder structure: notebook names don't match (`Automobile Demand Forecasting .ipynb` has a trailing space)
- [ ] Replace the placeholder `user-images.githubusercontent.com/example/...` image links with real charts from `outputs/`
- [ ] Decide whether the 17 MB `data/train.csv` stays in git (or `.gitignore` it and add a download note)
- [ ] Delete the five stale weekly-plan branches listed above

## Later
- [ ] Turn on ruff for the notebooks too (fix the unused `numpy` import and duplicate imports in the automobile notebook)
- [ ] Streamlit dashboard for top-N forecasts
