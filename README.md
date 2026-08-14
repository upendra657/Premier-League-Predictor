# Premier League Decision Engine

A calibrated probabilistic forecasting system for English Premier League
fixtures, with a financial backtesting engine that measures whether those
forecasts are worth acting on.

The question this project answers is not *"who will win?"* — a bookmaker
already answers that better than almost any model. It is **"where is the
market wrong, by how much, and how much should you stake on it?"**

![Results](reports/results.png)

---

## Headline results

Walk-forward evaluated on **4,509 fixtures across 12 held-out seasons**
(2013/14 – 2024/25). The model never sees a season before predicting it.

| | Brier ↓ | Log loss ↓ | RPS ↓ | ECE ↓ | Accuracy |
|---|---|---|---|---|---|
| **Market (de-vigged closing line)** | **0.5634** | **0.9527** | **0.1941** | 0.0105 | **55.7%** |
| Random Forest + isotonic | 0.5762 | 0.9809 | 0.1996 | **0.0093** | 54.3% |
| XGBoost (natively calibrated) | 0.5792 | 0.9789 | 0.2006 | 0.0171 | 53.9% |
| Random Forest (uncalibrated) | 0.5902 | 0.9929 | 0.2033 | 0.0212 | 51.5% |

**The market is sharper. The model is better calibrated.** Those are different
properties, and the distinction is the whole basis of the strategy: the market
discriminates outcomes better on average, but the model's stated confidence is
more trustworthy, which is what staking decisions actually depend on.

### What calibration bought

| Estimator | Brier | ECE | Verdict |
|---|---|---|---|
| Random Forest | 0.5902 → 0.5762 (**−2.4%**) | 0.0212 → 0.0093 (**−56.3%**) | Isotonic calibration is essential |
| XGBoost | 0.5792 → 0.5808 (+0.3%) | 0.0171 → 0.0172 (+0.2%) | Already calibrated; post-hoc fitting adds noise |

Random Forest probabilities are vote proportions, not likelihoods, and are
badly miscalibrated out of the box. XGBoost trained against `mlogloss` is
optimising a proper scoring rule directly, so it arrives calibrated and
isotonic regression has nothing left to fix. **Calibration is a targeted
remedy, not a universal improvement** — a finding this project measures rather
than assumes.

### Economic results

| Scenario | Bets | ROI / unit staked | 95% CI | p(ROI ≤ 0) | Bankroll |
|---|---|---|---|---|---|
| Line-shopped, 12 seasons | 2,154 | **+7.2%** | [+1.3%, +13.4%] | 0.007 | £1,000 → £7,328 |
| Single bookmaker, 12 seasons | 2,166 | +3.2% | [−2.4%, +9.2%] | 0.125 | £1,000 → £1,838 |
| **Strict holdout** (band refitted on 2013/14–18/19) | 877 | +1.5% | [−9.2%, +12.7%] | 0.379 | £1,000 → £1,035 |

Read those three rows together, because they tell one story:

1. **Execution is roughly half the edge.** A single bookmaker's line carries a
   **5.31% overround**; taking the best price across books cuts it to **0.35%**.
   That 5-point swing is larger than any modelling gain in this project.
2. **The full-period edge is statistically real** (season-clustered bootstrap,
   p = 0.007) but rests on a selection band chosen with hindsight over the
   whole sample.
3. **Under strict discipline the edge is not significant.** Refit the band on
   the first six seasons and apply it blind to the last six, and ROI falls to
   +1.5% with a confidence interval straddling zero.

The honest conclusion is that this system finds a **marginal, unstable edge**
in a market that is close to efficient. That is the correct scientific finding,
and reporting it is more useful than a tuned number that would not survive
contact with a real bankroll.

### Where the edge actually is

| Model vs market (relative edge) | Bets | Hit rate | Return |
|---|---|---|---|
| < 0% | 6,889 | 33.9% | −1.3% |
| 0 – 5% | 1,129 | 40.3% | −3.6% |
| 5 – 10% | 1,003 | 37.6% | −3.9% |
| **10 – 20%** | **1,581** | **37.9%** | **+10.2%** |
| **20 – 40%** | **1,672** | **31.2%** | **+2.6%** |
| > 40% | 1,253 | 17.9% | −4.7% |

Profit lives in a **band**, not a tail. Small disagreements with the market are
noise that cannot clear the overround; enormous disagreements are the model
being overconfident on longshots. Naive "bet everything with positive EV"
systems lose money precisely because they treat those three regimes as one.

---

## Architecture

```
src/
├── config.py      Typed configuration; every hyper-parameter in one place
├── data.py        Ingestion, team-name reconciliation, odds join, de-vigging
├── features.py    Elo, shot-quality xG proxy, decayed rolling form, fatigue
├── train.py       XGBoost + RF, CalibratedClassifierCV, walk-forward scoring
├── backtest.py    EV detection, band calibration, fractional Kelly simulation
├── report.py      Consolidated results bundle
├── plots.py       Result figures
└── main.py        FastAPI inference service
tests/             24 tests, weighted toward leakage and staking correctness
Dockerfile         Multi-stage build, non-root runtime, healthcheck
```

### Feature engineering

**Dynamic Elo.** Ratings update after every fixture in a single chronological
pass. The K-factor is scaled by a margin-of-victory multiplier
(`ln(|GD|+1) · 2.2/(0.001·Δrating + 2.2)`) that rewards big wins with
diminishing returns while damping rating inflation when a strong favourite
wins heavily. A 65-point home-advantage offset enters the expectancy curve, and
ratings regress 25% toward the league mean between seasons to model squad
turnover and promotion.

**Shot-quality expected goals.** The dataset carries no true xG, so a Poisson
regression maps each team's shot profile (volume, shots on target, corners,
accuracy) onto expected goals. Fitted **only on the burn-in season**, which is
excluded from training and evaluation. Shots on target dominate the fit, as
they should. This recovers the quantity xG is for: goals deserved from chances
created, stripped of finishing luck.

**Time-decayed rolling form.** Rolling xG, xGA and points over a 5-match window
with exponential decay (2.5-match half-life), so a fixture five games ago
carries a quarter of the weight of the most recent one. Implemented as a fixed
linear combination of lagged series — fully vectorised, no `rolling.apply`.

**Fatigue.** Rest-day differential, clipped at 14 days so summer breaks do not
swamp the signal.

### Leakage policy

Every feature is computed strictly from information available before kick-off:

- Elo ratings are snapshotted *before* the fixture is scored, then updated.
- Rolling windows are lagged one match; a team's own current result is never
  visible to it.
- Rolling state never crosses between teams.
- The shot-quality model touches only burn-in seasons.
- Walk-forward training uses only seasons that finished before the target one.
- Calibration folds are chronological (`TimeSeriesSplit`), so the calibrator is
  never fitted on data postdating its validation slice.

Nine of the 24 tests exist purely to enforce these properties.

---

## Quickstart

```bash
pip install -r requirements-dev.txt

python -m src.data        # ingest + join odds (downloads once, then cached)
python -m src.features    # build the feature store
python -m src.train       # walk-forward evaluation + production artifacts
python -m src.backtest    # strategy calibration + held-out performance
python -m src.report      # consolidated results bundle
python -m src.plots       # figures

pytest -q                 # 24 tests
```

### Serving

```bash
uvicorn src.main:app --reload
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | Liveness + whether the model artifact loaded |
| `GET /model` | Metadata card including walk-forward metrics |
| `POST /predict` | Calibrated Home / Draw / Away probabilities |
| `POST /value` | Probabilities + edge + Kelly stake against supplied odds |

```bash
curl -X POST localhost:8000/value -H 'Content-Type: application/json' -d '{
  "fixture": {"home_team":"Arsenal","away_team":"Chelsea",
              "elo_home":1650,"elo_away":1580,
              "xg_home_roll":1.9,"xga_home_roll":0.9,
              "xg_away_roll":1.4,"xga_away_roll":1.2,
              "points_home_roll":2.2,"points_away_roll":1.5,
              "rest_days_home":7,"rest_days_away":3,"matchweek":12},
  "odds": {"home":2.10,"draw":3.60,"away":3.40}
}'
```

Returns per-selection edge, expected value, Kelly stake fraction and a
`recommended` flag — selections outside the profitable band are returned with a
zero stake rather than silently dropped.

### Container

```bash
docker build -t plde:2.0 .
docker run -p 8000:8000 plde:2.0
```

Multi-stage build; the runtime image carries the virtualenv, `src/` and
`artifacts/` only — training dependencies and raw data never ship. Runs as an
unprivileged user with a `/health` healthcheck.

---

## Data

| Source | Role |
|---|---|
| Curated EPL results, 2000/01–2024/25 (9,380 fixtures) | Match spine: outcomes, shots, corners, cards |
| [football-data.co.uk](https://www.football-data.co.uk/) mirror | 1X2 closing odds, average and best-of-market |

Odds join covers **99.1%** of fixtures. The join is validated, not assumed:
ingestion raises if coverage drops below 95%, because a silent team-name
mismatch would otherwise produce a meaningless backtest rather than an error.

---

## Known limitations

- **The strict-holdout edge is not statistically significant.** Treat +7.2% as
  an upper bound obtained with hindsight, and +1.5% as the honest expectation.
- **xG is a proxy.** Match-level shot-quality regression is not shot-level xG;
  real Understat/StatsBomb xG would likely sharpen the rolling features.
- **Closing odds assume ideal execution.** Real staking faces limits, line
  movement and account restriction. The single-book row is the pessimistic bound.
- **Rolling form carries a venue-alternation bias.** Because fixtures alternate
  home and away, a team's decayed form entering a home match is systematically
  lower than entering an away match (observed in 44 of 46 clubs). The effect is
  a near-constant offset the tree models absorb, but venue-split form would be
  a cleaner encoding.
- **No player-level data.** Injuries, suspensions and rotation are unmodelled
  and are plausibly the largest remaining source of signal.

---

## License

MIT.
