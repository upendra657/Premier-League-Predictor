# What this project actually is

A plain-English map of the Premier League Predictor, written so you can hold the
whole thing in your head. No jargon that isn't explained the first time it appears.

Everything below was read off the code and the stored results, not from memory.

---

## 1. The short version

You have a machine that looks at a football match **before it is played**, and
says how likely each of the three outcomes is:

> Liverpool vs Man City → **60.4% home win, 18.7% draw, 20.8% away win**

It does this by learning from 25 seasons of past matches. Then it does one extra
thing that most such projects don't: it checks its answer against what bookmakers
were charging at the time, and asks *"do I disagree with the market, and would
acting on that disagreement have made money?"*

That's it. That's the whole project. Everything else is plumbing that makes those
two steps trustworthy.

---

## 2. Follow one real match through the machine

This is the best way to understand it. Here is an actual fixture from your data,
with the actual numbers the system produced.

**Liverpool vs Manchester City, 1 December 2024, matchweek 13.**

### Step 1 — The raw row arrives

From `data/raw/epl_final.csv`, which is your file of 9,380 completed matches:

| | |
|---|---|
| Date | 2024-12-01 |
| Teams | Liverpool (home) vs Man City (away) |
| Final score | 2–0 |
| Shots, shots on target, corners, fouls, cards | for both sides |

Note what's here and what isn't. There are **no players**, no lineups, no injuries,
no minutes. The unit of your data is the *match*, never the person.

### Step 2 — A second file is joined on

From a public mirror of football-data.co.uk, the system downloads what bookmakers
were offering. For this match: **2.09 on Liverpool, 3.70 on the draw, 3.37 on City.**

This is a completely separate data source from your CSV. Your file has no odds at
all. Without this second file there is no market comparison, no backtest, and no
project as it currently exists.

### Step 3 — The machine works out what it knew *at that moment*

This is the important bit, and the part most projects get wrong. To predict a match
honestly you may only use information that existed *before kickoff*. So the system
rewinds and computes eighteen numbers:

**How good is each team? (Elo)**
```
elo_home            1684.8      Liverpool's running strength score
elo_away            1683.6      City's
elo_diff               1.2      almost nothing between them
elo_win_expectancy     0.594    with home advantage, Liverpool ~59% to win
```

Elo is a single number per team that goes up when you win and down when you lose.
Beat someone much better than you and it jumps a lot; beat someone terrible and it
barely moves. It's the same system chess uses. Yours also scales the update by how
heavily you won, and pulls every team 25% back toward average between seasons
(because squads change over the summer).

**How have they been playing lately? (rolling form)**
```
xg_home_roll        1.703       Liverpool creating ~1.7 "expected goals" a game
xg_away_roll        1.137       City only 1.14
xga_home_roll       0.970       Liverpool conceding under 1
xga_away_roll       1.344       City conceding 1.34
points_home_roll    2.719       Liverpool taking 2.7 points per game
points_away_roll    0.741       City taking 0.74 — dreadful
form_diff           1.978       the gap
```

"Rolling" means the last five matches. But not weighted equally — the most recent
game counts about four times as much as the game five back. That's the "time decay."

**Everything else**
```
rest_days_home      7.0         days since Liverpool last played
rest_days_away      8.0
rest_diff          -1.0
matchweek          13.0         where we are in the season
```

Look at that form gap. City were, at that moment, in genuine crisis, and the
machine could see it.

### Step 4 — The model gives its answer

Those 18 numbers go into XGBoost — think of it as a very large stack of
yes/no questions ("is elo_diff above 40? is form_diff below 0.5?") learned from
thousands of past matches. Out comes:

> **Home 60.4% · Draw 18.7% · Away 20.8%**

### Step 5 — Compare with the market

The bookmaker's odds imply **45.8% / 25.9% / 28.4%**.

(Those percentages don't come straight from the odds. Raw bookmaker odds always
add up to more than 100% — here 104.55% — and that extra 4.55% is the bookmaker's
margin, the "overround." The system strips it out to get a fair comparison. That's
what "de-vigging" means.)

So the model thought Liverpool were **60.4%** and the market thought **45.8%**.
That's a disagreement of 14.6 percentage points, or a **32% relative edge**.

### Step 6 — Decide whether to act

The system has a pre-set rule: only act when the relative edge is between 10% and
40%, the odds are between 1.80 and 5.00, and the absolute gap is at least 3 points.
This one qualifies on all counts. So it records a hypothetical bet on Liverpool at 2.09.

Why a *band* and not just "any edge"? Because your own results showed that small
disagreements are noise that can't clear the bookmaker's margin, and huge
disagreements are usually the model being wrong about a longshot. The money lives
in the middle. That finding is one of the better things in this repo.

### Step 7 — Reality arrives

Liverpool won 2–0. The model was right, the market was wrong, the bet won.

That happens 35.7% of the time across 2,154 such bets — which sounds bad until you
notice the average odds were 3.22, so you only need to be right about 31% of the
time to break even.

---

## 3. What every file does

### The pipeline (`src/`)

| File | Lines | What it does, plainly |
|---|---|---|
| `config.py` | 218 | Every setting in one place. K-factor, home advantage, window sizes, bet limits. No magic numbers hidden in the code. |
| `data.py` | 306 | Loads your CSV, downloads the odds file, glues them together, strips the bookmaker margin. Refuses to continue if fewer than 95% of matches get odds. |
| `features.py` | 499 | The heart. Turns raw match rows into the 18 numbers above. Elo, the expected-goals estimate, rolling form, rest days. |
| `train.py` | 366 | Trains the model season by season, scores it properly, and saves the winner. |
| `backtest.py` | 486 | The betting simulation. Picks bets, sizes them, settles them day by day, and works out confidence intervals. |
| `report.py` | 189 | Boils everything down to the numbers that get published. |
| `main.py` | 438 | A web API — you send it two team names, it sends back probabilities. |
| `dashboard.py` | 302 | Builds a standalone HTML results page. |
| `workbook.py` | 663 | Builds an Excel file with 29,332 live formulas over the bet ledger. |
| `export_web.py` | 125 | Converts the model into JSON so it can run inside a browser. |
| `plots.py` | 167 | The charts for the README. |

### The data (`data/`)

- `raw/epl_final.csv` — your 9,380 matches, 2000/01 to 2024/25
- `raw/market_odds.csv` — the downloaded bookmaker odds
- `processed/feature_store.parquet` — 9,380 rows × 56 columns, the fully-built table
- `processed/oos_predictions.parquet` — **22,545 rows.** Every prediction the system
  made on matches it had never seen, for all five model variants. This file is
  more valuable than you probably realise; see §6.

### The outputs (`reports/`, `artifacts/`)

The trained model (1.8MB), a team-strength snapshot, an HTML dashboard, a 2.2MB
Excel workbook, a browser-runnable copy of the model, and seven CSVs of results.

### The tests (`tests/`)

42 of them. 29 on the pipeline, 13 on the API. Nine exist for one purpose only:
to fail if information from *after* a match ever leaks into the features used to
predict it. One of them literally checks that a five-match average doesn't
accidentally include the match it's predicting.

---

## 4. The numbers that are actually yours

Read straight out of `reports/results.json`. These are real.

**The data**
```
9,380 matches · 25 seasons · 8,923 usable · 99.12% got odds attached
```

**How well it forecasts** — tested on 4,509 matches across 12 seasons it never
trained on. Lower is better for the first four columns.

| | Brier | Log loss | RPS | Calibration error | Accuracy |
|---|---|---|---|---|---|
| **Bookmakers** | **0.5634** | **0.9527** | **0.1941** | 0.0104 | **55.7%** |
| Random Forest + isotonic | 0.5762 | 0.9808 | 0.1996 | **0.0093** | 54.3% |
| **XGBoost (the one you ship)** | 0.5792 | 0.9789 | 0.2006 | 0.0171 | 53.9% |
| XGBoost + isotonic | 0.5808 | 0.9797 | 0.2014 | 0.0172 | 53.8% |
| XGBoost + sigmoid | 0.5816 | 0.9791 | 0.2021 | 0.0171 | 54.0% |
| Random Forest raw | 0.5902 | 0.9929 | 0.2033 | 0.0212 | 51.5% |

**Read this honestly: the bookmakers beat you on every measure except calibration.**
That is not a failure. Bookmakers have team news, injury reports, and millions of
pounds of other people's opinions moving their prices. Getting within 3% of them
using shot counts and a rating system is genuinely good. And you beat them on the
one thing you optimised for.

*What the columns mean:* **Brier** — were your percentages close to what happened.
**Log loss** — same idea, but punishes confident wrongness brutally. **RPS** —
like Brier but knows a draw sits between a home and away win, so predicting
"away" when it was a home win is worse than predicting "draw." **Calibration
error** — when you say 60%, does it happen 60% of the time. **Accuracy** — how
often your top pick was right, which you deliberately did not optimise for.

**The betting simulation**

| | Bets | Kelly ROI | Flat-stake ROI | 95% range | Worst drawdown |
|---|---|---|---|---|---|
| Full period, best odds | 2,154 | +4.8% | **+7.2%** | +1.3% to +13.4% | −47% |
| Full period, one bookmaker | 2,166 | — | — | — | — |
| **Strict holdout** | 877 | +0.3% | **+1.5%** | −9.2% to +12.7% | −54% |

**A labelling trap worth understanding.** The famous "+7.2%" is the **flat-stake**
return, not the Kelly one. Kelly returned **+4.8%**. Both are in `results.json`.

This is not an error in the numbers — it's deliberate, and correct. `backtest.py`
bootstraps the confidence interval on `unit_return`, the profit per unit staked.
With a constant stake, ROI on turnover *is* the mean unit return, so the CI
(+1.3% … +13.4%) and p-value belong to the flat-stake figure and are properly
paired with it. Flat staking isolates **how good the bet selection was** from
**how the money was sized** — which is the right way to measure selection.

`LIMITATIONS.md` labels the column "Flat ROI" and is correct. The README's table
just says "ROI", and one resume bullet describes it as "a fractional-Kelly
strategy returned +7.2%", which it didn't. That sentence is the only real mistake,
and it's a one-line fix.

The holdout row is the honest one. When you refit the bet-selection rule on early
seasons only and applied it blind to later ones, the edge mostly evaporated
(p = 0.379 — meaning there's a 38% chance you'd see a result this good from pure
luck). Six of twelve seasons lost money.

---

## 5. The five genuinely clever things

These are your interview material. You have five real findings. Most portfolio
projects have none.

**1. You lose to the market, and that's the thesis.**
The market forecasts better than you (0.5634 vs 0.5792 Brier). But *you are better
calibrated* than the raw alternative, and calibration — not raw accuracy — is what
matters when you're deciding whether a price is wrong. Understanding the difference
between "ranks outcomes well" and "gets the percentage right" is a genuinely
advanced distinction and you can defend it.

**2. Calibration is a targeted fix, not a free upgrade.**
Isotonic regression cut Random Forest's calibration error by **56%** (0.0212 →
0.0093). Applied to XGBoost it did *nothing* — actually made Brier slightly worse
(0.5792 → 0.5808) — because XGBoost's training objective already optimises for
this. You reported the null result instead of hiding it.

**3. Returns are non-monotonic in edge.**
Below 10% relative edge you lose. Between 10% and 40% you win. Above 40% you lose
again. "Bigger disagreement with the market = better bet" is intuitive and wrong,
and you have the numbers showing three different regimes.

**4. Execution beats modelling.**
Shopping for the best price across bookmakers cuts the margin from 4.49% to 0.35%
on the same 7,570 matches. That single operational choice is worth more than every
modelling improvement in the project combined. It's a great "the boring thing
mattered most" story.

**5. The best-scoring model is the worst decision-maker.**
Random Forest + isotonic wins on Brier, RPS *and* calibration error — and loses
18.5% of turnover. The XGBoost you actually ship scores worse and makes money.
Why: calibration is measured across *all* predictions, but you only bet on the
small tail where you disagree with the market, and isotonic regression flattens
exactly those disagreements (279 qualifying bets instead of 877). This is the best
thing in the project. "The model with the best score was the wrong model to deploy,
and here's the mechanism" is a senior-level observation.

---

## 6. What's broken, half-done, or not what it says

Honest list. None of it is hard to fix.

**The expected-goals number is a stand-in, and it's wrong at the edges.**
Real expected goals is computed per shot, from where on the pitch it was taken.
You don't have shot locations, so the system fits a formula on shot counts, shots
on target and corners. It works well enough as a model input, but as a published
number it's badly compressed: it says Liverpool "deserved" 52.0 goals when they
scored 81, and Southampton "deserved" 32.1 when they scored 25. It flatters bad
teams and undersells good ones. Fine as an ingredient; unusable as a headline.

**The rest-day feature is documented as something it isn't.**
It's labelled a fatigue metric. The model learned the *opposite* of fatigue: more
rest slightly *reduces* home win chance. The raw data agrees (50.0% home wins on
0–3 days' rest vs 44.9% on 11–14). And it isn't a hidden team-quality effect —
the correlation between rest days and Elo is −0.007, essentially zero. So the
signal is real and the explanation attached to it is wrong. It also ranks 16th,
17th and 18th of 18 features by importance, so it barely matters either way.

**The API and Docker have never been run.**
`main.py` has four working endpoints and `Dockerfile` is written, but `docker build`
has never been executed and nothing is deployed anywhere. Your resume says
"containerised FastAPI service." The code is real; the claim isn't yet.

**The data stops in May 2025.**
And 2024/25 only has 350 of its 380 matches. Since then a whole season has finished
and 2026/27 has kicked off. Anything the project says about "now" means "as of
eighteen months ago."

**Small stuff.** A 2.2MB generated Excel file is committed to git. An `httpx2`
deprecation warning fires on every test run.

**And one asset you may not know you have.**
`data/processed/oos_predictions.parquet` holds **22,545 predictions** — 4,509
matches × 5 model variants — every one of them made on a match the model had never
seen. That is a ready-made, honest track record sitting on your disk. Any page
that says "here's what we predicted and here's what happened" can be built from
this file today, with no new data and no fixes to anything above.

---

## 7. So what *is* this, really?

**It is:** a carefully built forecasting system with unusually good discipline
around not cheating, evaluated the way a statistician would evaluate it, with five
real findings and a large stock of honest out-of-sample predictions.

**It is not:** a live service, a product with users, a thing that knows about
players, or a system that knows what happened after May 2025.

**The single most misleading thing about it right now** is the front door. It
presents as a betting system, which is the least interesting thing it does and the
hardest to talk about. Underneath is a much better project about how to know
whether a number can be trusted — which is a question worth caring about whether
or not you like football.

That's the gap between what this is and how it reads. Closing it is mostly writing,
not code.
