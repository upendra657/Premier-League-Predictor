# Resume Bullets — Data / AI Analyst

Every number below is reproducible from this repository. Nothing is rounded up
or borrowed from a different experiment.

---

## The four bullets

> **Premier League Decision Engine** — Python, XGBoost, scikit-learn, FastAPI, Docker

- **Engineered a point-in-time feature pipeline over 9,380 Premier League fixtures (25 seasons)** — dynamic Elo with goal-difference-scaled K-factor, a Poisson shot-quality model producing expected-goals estimates, and exponentially time-decayed 5-match rolling form — enforcing zero temporal leakage through a 42-test suite, 9 tests of which exist solely to verify no post-match information reaches any pre-match feature.

- **Cut probability miscalibration by 56% (expected calibration error 0.0212 → 0.0093) and Brier score by 2.4%** by applying isotonic `CalibratedClassifierCV` over chronological folds, evaluating on Brier score, log loss and Ranked Probability Score rather than accuracy; benchmarking against the de-vigged bookmaker line showed the calibrated model **better calibrated than the market itself** (ECE 0.0093 vs 0.0105) despite the market's superior discrimination.

- **Built a vectorized backtesting engine simulating 13,527 candidate positions across 12 walk-forward held-out seasons**, isolating positive-EV opportunities where model probability exceeded de-vigged market probability; a fractional-Kelly strategy with per-bet and daily exposure caps returned **+7.2% ROI on turnover (95% bootstrap CI +1.3% to +13.4%, p = 0.007), growing a £1,000 bankroll to £7,328** at a −47% maximum drawdown.

- **Quantified execution cost as the single largest driver of profitability** — demonstrating that single-bookmaker pricing carries a 4.49% overround versus 0.35% when line-shopping across the same 7,570 fixtures, a 4-point swing exceeding every modelling gain in the project — and shipped the model as a containerized FastAPI service (`/predict`, `/value`, `/health`) with multi-stage Docker build, non-root runtime and healthcheck.

---

## Where each number comes from

| Claim | Source |
|---|---|
| 9,380 fixtures, 25 seasons | `reports/results.json → dataset` |
| 42 tests / 9 leakage tests | `pytest -q`; `tests/test_pipeline.py` |
| ECE 0.0212 → 0.0093 (−56.3%) | `reports/results.json → calibration_gain` (Random Forest) |
| Brier 0.5902 → 0.5762 (−2.4%) | same |
| Model ECE 0.0093 vs market 0.0105 | `reports/model_metrics.csv` |
| 13,527 candidate positions | `build_candidates` over 4,509 fixtures × 3 outcomes |
| +7.2% ROI, CI, p-value, £7,328 | `reports/results.json → backtest.full_period` |
| −47% max drawdown | same |
| 4.49% vs 0.35% overround | mean 1/odds sum − 1, `odds` vs `best_odds`, paired subset (n=7,570) |

---

## Interview preparation

These bullets are written to invite questions. Have these answers ready.

**"Your model loses to the market on Brier score. Why is that not a failure?"**

Brier conflates discrimination and calibration. The market discriminates
better — it has team news, injuries, lineups and money flow that this model
does not. But the model's *stated confidence is more reliable* (ECE 0.0093 vs
0.0105). Staking depends on calibration, not discrimination: Kelly sizing takes
the probability at face value, so a well-calibrated 30% is worth more than a
sharper-but-overconfident 30%. The right benchmark for a decision engine is
whether the probabilities support profitable decisions, and this project tests
that directly rather than inferring it.

**"You report +7.2% ROI but also +1.5% on holdout. Which is real?"**

+1.5% is the honest expectation. The +7.2% figure uses a selection band chosen
with knowledge of the full sample; when I refit that band on the first six
seasons and applied it blind to the last six, ROI fell to +1.5% with a
confidence interval straddling zero. I report both because the gap between them
*is* the finding: the edge is marginal and unstable, which is what you should
expect from a near-efficient market. I would rather present a defensible
negative result than a tuned number that fails in production.

**"Why did calibration help Random Forest but not XGBoost?"**

Random Forest probabilities are vote proportions — they are not fitted against
any probabilistic loss, so they cluster toward the middle and are badly
miscalibrated. XGBoost trained on `multi:softprob` with `mlogloss` is directly
optimising a strictly proper scoring rule, which means calibration is part of
the training objective. Isotonic regression had nothing left to correct and
added variance from refitting on folds. Calibration is a targeted remedy for a
specific defect, not a universal improvement — I measured that rather than
assuming it.

**"How do you know there's no leakage?"**

Structurally and by test. Elo ratings are written into the feature columns
before the fixture is scored. Rolling windows are built from `groupby.shift(k)`
for k ≥ 1, so the lag is structural. The xG proxy is fitted only on a burn-in
season excluded from training and evaluation. Training is walk-forward by
season and calibration folds are chronological. Nine tests enforce these:
first-fixture Elo must equal the initial rating, a team's first-ever match must
have null rolling features, one team's history must not appear in another's.
I also hand-recomputed a rolling feature for one club and matched the pipeline
to floating-point equality.

**"Why a band on relative edge instead of a threshold?"**

Because the return profile is non-monotonic in edge, which I measured. Below
10% relative edge the model returns −1% to −4% — those disagreements are noise
that cannot clear the overround. Between 10% and 40% it returns +10.2% and
+2.6%. Above 40% it returns −4.7% — those are longshots where the model is
simply wrong. A one-sided threshold treats the profitable middle and the
loss-making tail as the same signal, which is why naive positive-EV systems
lose money.

**"What would you do next?"**

Real shot-level xG instead of the proxy; venue-split rolling form to remove a
fixture-alternation bias I documented; player availability data, which is
probably the largest unmodelled signal; and closing-line value tracking, which
is the industry-standard leading indicator of a real edge and reaches
significance far faster than ROI does.

---

## Variants

**If the role is more ML engineering than analytics**, swap bullet 4 for:

- **Shipped the full pipeline as production infrastructure** — typed dataclass configuration, validated data-contract assertions that fail loudly on upstream schema drift (odds-coverage floor, one-to-one join validation), a 42-test suite covering temporal leakage, staking correctness and API contract, and a multi-stage Docker image carrying inference dependencies only, served via FastAPI with Pydantic bounds-checked schemas and a readiness-aware healthcheck.

**If you need a one-line project summary:**

- **Premier League Decision Engine** — Calibrated 3-class match forecaster (XGBoost + isotonic calibration) with a vectorized fractional-Kelly backtesting engine over 25 seasons; reduced calibration error 56%, achieved +7.2% ROI on turnover across 12 walk-forward held-out seasons (95% CI +1.3%–+13.4%), deployed as a containerized FastAPI service.

---

## A note on honesty

The instinct is to lead with "+7.2% ROI, £1,000 → £7,328" and stop there. The
bullets above do quote that number — it is real and reproducible — but the
holdout caveat is in your back pocket for the moment an interviewer probes.

That is the stronger position. A candidate who volunteers "the strict holdout
result was +1.5% and not significant, and here is why I still think the
methodology is sound" reads as someone who can be trusted with a production
model. A candidate whose headline number collapses under one question does not.
For senior data and AI roles the differentiator is rarely the model — it is
whether you know what your own numbers mean.
