# Implementation Plan

The order below is the order the work was actually done, and the order to
re-do it in. Each phase has an exit criterion — a check that must pass before
the next phase starts. Most failed portfolio ML projects skip phase 0 and
discover in phase 4 that the target was unreachable from the available data.

---

## Phase 0 — Audit what the data can actually support

**Before writing any modelling code**, list every feature and metric the design
calls for and confirm the data exists.

The original repository could not support the stated design:

| Requirement | Status in original repo | Resolution |
|---|---|---|
| Rolling xG / xGA | **No xG anywhere** | Poisson shot-quality proxy from shots/SoT/corners |
| Backtesting vs bookmaker odds | **No odds anywhere** | Join football-data.co.uk 1X2 closing lines |
| 3-class H/D/A model | Target pre-collapsed to binary `H`/`NH` | Rebuild target from raw results |
| Point-in-time features | Pre-computed columns of unclear provenance | Recompute every feature from raw results |

**Exit criterion:** every feature in the design maps to a column that exists,
or to a documented derivation from columns that exist.

---

## Phase 1 — Configuration and project skeleton

Create `src/config.py` first. Every hyper-parameter — Elo K-factor, decay
half-life, Kelly fraction, edge band — lives in typed frozen dataclasses.

Why first: the alternative is magic numbers scattered across five modules,
which makes experiments unreproducible and code review impossible.

**Exit criterion:** no numeric literal in any pipeline module that a reviewer
would want to change.

---

## Phase 2 — Ingestion (`src/data.py`)

1. Load the curated results file as the authoritative **match spine**. Validate
   required columns, parse dates, canonicalise team names.
2. Assign each fixture a stable `match_id` (`date_home_away`) — every
   downstream join keys on this.
3. Download 1X2 odds from the public mirror, slice to division `E0`, cache to
   disk so reruns are offline.
4. Join odds onto the spine with `validate="one_to_one"`.
5. **Assert coverage.** Raise if under 95% of fixtures receive odds.
6. De-vig: normalise implied probabilities by the booksum to recover the
   market's fair estimate, and retain the overround as its own column.

The coverage assertion is the load-bearing step. A team-name mismatch
("Nottm Forest" vs "Nott'm Forest") silently drops rows; without the assertion
the backtest still runs and still reports a number, and the number is garbage.

**Exit criterion:** ≥ 95% odds coverage; de-vigged probabilities sum to 1.0;
mean overround is a plausible 4–6%.

---

## Phase 3 — Feature engineering (`src/features.py`)

Build in this order, because each stage depends on the previous one's shape.

**3a. Elo.** One chronological pass. Read ratings into the feature columns
*before* scoring the fixture, then update. Apply the margin-of-victory
multiplier and between-season regression to the mean.

**3b. Team-perspective reshape.** Convert one row per fixture into two rows,
one per participating team. Every rolling computation then becomes a single
`groupby("team")` rather than parallel home/away logic that can drift apart.

**3c. Shot-quality xG proxy.** Fit a Poisson regression on the **burn-in season
only**, predicting goals from shots, shots on target, corners and accuracy.
Score every team-match for xG (own shot profile) and xGA (opponent's).

**3d. Decayed rolling windows.** Weighted mean over the previous 5 matches with
an exponential half-life. Implement as `Σ wₖ·x.shift(k) / Σ wₖ·mask.shift(k)` —
vectorised, and the `shift` makes the one-match lag structural rather than
something a future edit can accidentally remove.

**3e. Fatigue and derived terms.** Rest-day differential; matchup interactions
(`xg_matchup_edge` = what the home side creates against what the away side
concedes, minus the mirror image).

**3f. Pivot back** to the fixture grid and flag `is_burn_in` / `is_modellable`.

**Exit criterion:** hand-recompute one team's rolling feature for one fixture
and match the pipeline to floating-point equality. This catches misaligned
merges that summary statistics hide.

---

## Phase 4 — Training and calibration (`src/train.py`)

**Metrics before models.** Implement multiclass Brier, log loss, Ranked
Probability Score and Expected Calibration Error first, so there is never a
temptation to report accuracy because it was the metric already available.

**Walk-forward, not cross-validation.** For each evaluation season, train on
every prior season only. Random k-fold on time-series data leaks the future
into the past and inflates every metric.

**Calibrate with chronological folds.** `CalibratedClassifierCV` with
`cv=TimeSeriesSplit(5)`, both isotonic and sigmoid, over both base estimators.

**Always align class order.** Never assume `predict_proba` column *i* is class
*i*; reindex explicitly against `estimator.classes_`.

**Benchmark against the market** on exactly the same fixtures. Without this the
metrics have no scale — a Brier of 0.58 is meaningless until you know the
market scores 0.56 and a base-rate forecast scores 0.62.

**Exit criterion:** calibrated variants evaluated against uncalibrated ones and
against the market, on identical fixtures.

---

## Phase 5 — Decision engine (`src/backtest.py`)

1. **Explode** each fixture into three candidate bets.
2. **De-vig per match**, compute absolute edge, relative edge and expected value.
3. **Filter to a band**, not a threshold. Use *relative* edge as the primary
   filter — an absolute 3pp edge on a 5% longshot is a 60% relative claim and is
   almost always model error, while 3pp on a 50% shot is a 6% claim.
4. **Size with fractional Kelly**, `(p·o − 1)/(o − 1)` scaled by 0.25 and capped
   per bet.
5. **Cap daily exposure.** Kelly is derived for sequential independent bets;
   Premier League fixtures cluster on Saturdays, so without a joint cap a dozen
   "small" positions compound into one large one and variance drag destroys the
   bankroll even with positive per-bet EV.
6. **Fit the band on early seasons, report on held-out ones.** A strategy tuned
   and scored on the same data is always profitable.
7. **Bootstrap the ROI interval, clustered by season.** Betting returns are
   heavy-tailed; a normal-approximation standard error badly understates
   uncertainty, and whole seasons run hot or cold together.
8. **Report both execution assumptions** — single-book and line-shopped.

**Exit criterion:** held-out ROI reported with a confidence interval, alongside
the in-sample number, with the gap between them stated plainly.

---

## Phase 6 — Service and container

`src/main.py`: load artifacts once in a `lifespan` handler, not per request.
Pydantic models with realistic bounds — `elo_home` capped at 2200 rejects
nonsense input at the edge rather than propagating it into the model. `/health`
reports whether the artifact actually loaded, so orchestrators can distinguish
"process alive" from "able to serve".

`Dockerfile`: multi-stage. Builder compiles wheels into a virtualenv; runtime
copies the venv, `src/` and `artifacts/` only. Non-root user, healthcheck.

**Exit criterion:** `/health` returns `model_loaded: true`, `/predict` returns
probabilities summing to 1.0, malformed input returns 422.

---

## Phase 7 — Verification

The tests that matter are the ones covering silent failure:

| Property | Why it matters |
|---|---|
| Elo read pre-update | Post-update ratings encode the result — total leakage |
| Rolling mean excludes current row | Off-by-one includes the outcome being predicted |
| Rolling state never crosses teams | A `groupby` mistake bleeds one club's form into another |
| Recent weighted above old | Verifies decay is applied, not just a flat mean |
| Probabilities sum to 1 | Class-order scrambling shows up here first |
| Brier = 0 perfect, 2 confidently wrong | Pins the metric's scale |
| RPS penalises distant errors more | Confirms ordinal awareness |
| Kelly matches closed form | Staking maths is where money is lost |
| Daily exposure cap holds | Guards the failure that wiped the first bankroll |
| Bankroll cannot go negative | Simulation must be physically possible |

**Exit criterion:** suite green, and each result in the README traceable to a
committed script that regenerates it.

---

## What to do next

1. **Real xG** from Understat or FBref, replacing the proxy (2014/15 onward).
2. **Venue-split rolling form**, removing the alternation bias documented in
   the README.
3. **Player availability** — injuries and suspensions are the largest
   unmodelled signal.
4. **Closing-line value tracking**, the industry-standard leading indicator of
   whether an edge is real, measurable long before ROI reaches significance.
5. **Ordinal or bivariate-Poisson targets** exploiting the H/D/A ordering
   directly rather than treating the classes as unrelated.
