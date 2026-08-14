# Rebuild as a calibrated decision engine with financial backtesting

Rewrites the project from a single-notebook classifier into a modular pipeline
that predicts calibrated probabilities, converts them into staking decisions,
and tests those decisions against real historical bookmaker prices.

The headline is not the accuracy. It is that the accuracy turned out not to be
the thing worth optimising.

## What changed

| | Before | After |
|---|---|---|
| Structure | one notebook | 11 modules under `src/`, 42 tests |
| Validation | random train/test split | walk-forward, refit each season |
| Metric | accuracy | Brier, log loss, RPS, ECE |
| Output | predicted label | calibrated probability vector |
| Evaluation | confusion matrix | ROI vs market with bootstrap CIs |
| Serving | Flask demo | FastAPI + multi-stage Docker |

## The three findings

**1. The model loses to the market, and that is the point.**
Walk-forward Brier is 0.5762 (Random Forest, isotonic) against 0.5634 for the
de-vigged bookmaker line. The market has team news, lineups and money flow this
model does not. But the model's *stated confidence* is better calibrated
(ECE 0.0093 vs 0.0104). Discrimination and calibration are different properties,
and Kelly staking depends on the second one — which is why an edge exists at all
despite the worse Brier score.

**2. Calibration is a targeted remedy, not a free win.**
Isotonic regression cut Random Forest's calibration error 56% and its Brier
score 2.4%. Applied to XGBoost it did nothing (+0.3% Brier), because `mlogloss`
is already a strictly proper scoring rule — XGBoost arrives calibrated, so
post-hoc fitting only adds fold noise. Reporting the null result matters more
than the positive one.

**3. Profit lives in a band, not a tail.**
Bucketing realised return by relative edge over the de-vigged price:

| Relative edge | Bets | Hit rate | Avg odds | ROI |
|---|---|---|---|---|
| < 0% | 6,889 | 33.9% | 3.92 | −1.3% |
| 0–5% | 1,129 | 40.3% | 3.34 | −3.6% |
| 5–10% | 1,003 | 37.6% | 3.61 | −3.9% |
| **10–20%** | **1,581** | **37.9%** | **3.77** | **+10.2%** |
| **20–40%** | **1,672** | **31.2%** | **4.55** | **+2.6%** |
| > 40% | 1,253 | 17.9% | 7.91 | −4.7% |

Small edges are noise; very large edges are the model being wrong about
longshots. Naive "bet everything with positive EV" loses money because it
treats all three regimes as one signal.

## Backtest, honestly reported

Fractional Kelly (25%, capped exposure), £1,000 bankroll, 2013/14–2024/25:

| Scenario | Bets | ROI | 95% CI | p(ROI≤0) | Bankroll |
|---|---|---|---|---|---|
| Line-shopped | 2,154 | +7.2% | +1.3% … +13.4% | 0.007 | £1,000 → £7,328 |
| Single bookmaker | 2,166 | +3.2% | −2.4% … +9.2% | 0.125 | £1,000 → £1,838 |
| Strict holdout | 877 | +1.5% | −9.2% … +12.7% | 0.379 | £1,000 → £1,035 |

The +7.2% uses a selection band chosen with hindsight over the whole sample.
Refit that band on 2013/14–2018/19 and apply it blind to the following six
seasons and ROI falls to +1.5%, with a confidence interval straddling zero.

**The honest read is a marginal, unstable edge in a near-efficient market.**
That gap between the in-sample and holdout number is reported here rather than
buried, because a backtest that only survives its own tuning is not a result.

Execution dominates modelling: over the 7,570 fixtures priced both ways,
single-book overround is 4.49% against 0.35% line-shopped. That 4-point swing
is larger than every modelling gain in this PR combined.

## Leakage control

Every rolling feature is shifted one fixture within team before aggregation, so
a match never contributes to its own inputs. Elo is recorded pre-match and
updated after. Nine of the 42 tests exist solely to assert this — including one
that shuffles class order to catch a `predict_proba` column mismatch, and one
that fails if a rolling mean includes its own row.

Date parsing is defended against Excel round-tripping, which rewrites ISO dates
into ambiguous `M/D/YY`; parsed dates are cross-checked against the independently
stated `Season` label and the loader raises if more than 1% fall outside their
season window.

## Reviewing this

- `src/features.py` — Elo, time-decayed rolling form, rest-day differential
- `src/train.py` — walk-forward harness, calibration, proper scoring rules
- `src/backtest.py` — de-vigging, edge selection, fractional Kelly, bootstrap CIs
- `tests/test_pipeline.py` — start here; the leakage tests document the design
- `reports/dashboard.html` — rendered results walkthrough
- `docs/IMPLEMENTATION_PLAN.md` — build phases and exit criteria

Reproduce with `python -m src.train && python -m src.backtest && python -m src.report`.
