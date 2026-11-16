# Smart Inventory Forecasting: Task List

_Last updated by the weekly review routine: 2026-11-16._

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

## Next step (pick this up first)
- [ ] **Open a PR from `feat/forecast-module` into `main` and merge it.** This is the blocker: `main`
      hasn't changed since 2025-11-10, so none of the module, tests or results show on the public repo.
      Takes ~5 minutes: review the diff on GitHub, then merge.
- [ ] **Make the notebook import from `forecast.py`** instead of re-defining the logic, so the notebook
      and the tested code can't drift apart.

## Housekeeping
- [ ] Make the notebook import from `forecast.py` instead of re-defining the logic
- [ ] Fix the README folder structure: notebook names don't match (`Automobile Demand Forecasting .ipynb` has a trailing space)
- [ ] Replace the placeholder `user-images.githubusercontent.com/example/...` image links with real charts from `outputs/`
- [ ] Decide whether the 17 MB `data/train.csv` stays in git (or `.gitignore` it and add a download note)
- [ ] Delete the five stale weekly-plan branches listed above

## Later
- [ ] GitHub Actions workflow that runs `pytest` on every push
- [ ] Streamlit dashboard for top-N forecasts
