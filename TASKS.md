# Smart Inventory Forecasting: Task List

_Last updated by the weekly review routine: 2026-11-25._

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
- [x] Pinned `requirements.txt` (prophet 1.5.0, cmdstanpy 1.3.2, pandas 3.1.1, numpy 2.4.6, pytest 9.2.0, ruff 0.17.6);
      CI installs the same pandas/pytest/ruff versions. Re-ran `scripts/compare_baseline.py` with the pins: output
      identical to the committed `baseline_comparison.csv`. Tests: 11 passed; ruff: clean. (2026-11-21)
- [x] README folder structure fixed: real notebook names, plus `forecast.py`, `scripts/`, `tests/`, `TASKS.md`,
      rendered as a code block so the tree no longer collapses into one line. (2026-11-21)
- [x] `test_build_forecast_summary_with_prophet_matches_committed_output`: runs the notebook's real Prophet path on
      store 1 / item 1 from `data/train.csv` and checks it against the committed `store1_top5_forecast_summary.csv` row
      (±0.05). Skips without Prophet, so CI is unaffected. Tests: 12 passed with Prophet; 10 passed, 2 skipped without. (2026-11-22)

- [x] Renamed `notebooks/Automobile Demand Forecasting .ipynb` to `notebooks/01_automobile_demand_forecast.ipynb`
      (no more trailing space) and updated the README tree. Tests: 10 passed, 2 skipped (no Prophet); ruff: clean. (2026-11-23)
- [x] `plot_forecast(history, forecast, path, title)` in `forecast.py` (matplotlib `Agg`, draws observed sales,
      forecast and the `yhat_lower`/`yhat_upper` band) plus `scripts/plot_forecasts.py`, which renders
      `outputs/store1_item1_forecast.png` and `outputs/store1_item15_forecast.png` from `data/train.csv`. The README's
      fake "Top5Items" image now shows these two real charts. New test `test_plot_forecast_writes_png` checks a valid PNG
      is written; matplotlib 3.11.2 pinned in `requirements.txt` and CI. Tests: 13 passed with Prophet; ruff: clean. (2026-11-24)

- [x] All README placeholder images replaced. `fit_forecast(..., return_model=True)` now also returns the fitted
      Prophet model; new `plot_components(model, forecast, path)` saves Prophet's trend/weekly/yearly chart.
      `scripts/plot_forecasts.py` now also writes `outputs/store1_item1_components.png` and
      `outputs/brake_pads_forecast.png` (history from `data/sales_data.csv`, forecast from the saved
      `Brake Pads_forecast.csv`). The components chart confirms the README's "Monday low, weekend high" claim.
      New test `test_plot_components_writes_png`. Tests: 14 passed with Prophet; ruff: clean. (2026-11-25)

## Next step (pick this up first)
- [ ] **Open a PR from `feat/forecast-module` into `main` and merge it.** This is still the blocker (carried over
      since October): `main` hasn't changed since 2025-11-10, so none of the module, tests or results show on the public repo.
      Takes ~5 minutes: review the diff on GitHub, then merge.
      The PR now gets a green/red CI check, so you can see the tests pass before merging.
- [ ] **Type-check `forecast.py` with mypy.** Small learning goal: `fit_forecast` now returns either a frame or a
      `(model, frame)` tuple, which is exactly the kind of thing a type checker should flag. Add `typing.overload`
      signatures for the two `return_model` cases, add `mypy` to `requirements.txt`, run `mypy forecast.py`, and
      add it to CI once it passes.

## Housekeeping
- [ ] Decide whether the 17 MB `data/train.csv` stays in git (or `.gitignore` it and add a download note)
- [ ] Delete the five stale weekly-plan branches listed above

## Later
- [ ] Turn on ruff for the notebooks too (fix the unused `numpy` import and duplicate imports in the automobile notebook)
- [ ] Streamlit dashboard for top-N forecasts
