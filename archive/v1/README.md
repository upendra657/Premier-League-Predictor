# v1 — the original project

This is the first version of the Premier League Predictor: a notebook-driven
classifier with a Flask demo, kept here so the project's evolution stays
visible and its history stays intact.

It is **not** the current codebase. Everything live now lives in `src/`, and
the top-level `README.md` describes it.

## What is here

| Path | What it was |
|---|---|
| `xgboost_model.py` | The original training script — load CSV, split randomly, fit, print accuracy |
| `logistic_regression_model.joblib`, `xgboost_model.json` | Fitted artifacts from that script |
| `Datasets/`, `Misc/` | Intermediate CSVs and confusion-matrix plots from the exploratory phase |
| `api/` | Flask service deployed to PythonAnywhere |
| `frontend/` | Static page that called that service |
| `models/` | Serialised model and feature list used by the Flask app |

## Why it was replaced

The v1 approach optimised accuracy on a random train/test split. Two problems:
a random split over time-ordered fixtures leaks future information into the
training set, and accuracy is indifferent to whether the predicted
probabilities mean anything — which matters as soon as you act on them.

The rebuild evaluates walk-forward by season, scores with proper scoring rules
(Brier, log loss, RPS) against the de-vigged bookmaker line, and tests the
probabilities financially through a fractional-Kelly backtest.

History is preserved through the move: `git log --follow archive/v1/<file>`
reaches the original commits.
