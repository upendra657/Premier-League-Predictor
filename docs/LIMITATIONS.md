# Known limitations

Every figure in this repository is reproducible, but reproducible is not the
same as robust. This document records what the results do *not* establish, and
where the project would break first under real use.

It exists because the headline number — **+7.2% ROI over twelve held-out
seasons** — is the least reliable thing here, and a reader who took it at face
value would be misled.

---

## 1. The strategy's edge does not survive out-of-sample selection

The +7.2% figure applies a selection band chosen with knowledge of the full
sample. Refit that band on 2013/14–2018/19 and apply it blind to the following
six seasons:

| Scenario | Bets | Flat ROI | 95% CI | p(ROI ≤ 0) |
|---|---|---|---|---|
| Line-shopped, full period | 2,154 | +7.2% | +1.3% … +13.4% | 0.007 |
| Single bookmaker | 2,166 | +3.2% | −2.4% … +9.2% | 0.125 |
| **Strict holdout, band refitted** | **877** | **+1.5%** | **−9.2% … +12.7%** | **0.379** |

The honest reading is a marginal, unstable edge in a near-efficient market.
The first row measures what the strategy returned over the period; the third
estimates what it would return going forward. Only the third is a forecast.

## 2. The served model is not the best-calibrated one

This is the most counter-intuitive result in the project, and the one most
likely to be read as an error.

| Variant | Brier ↓ | ECE ↓ | Holdout bets | Holdout ROI |
|---|---|---|---|---|
| **xgb_uncalibrated** (served) | 0.5792 | 0.0171 | 877 | **+1.5%** |
| xgb_sigmoid | 0.5816 | 0.0171 | 337 | −1.1% |
| rf_uncalibrated | 0.5902 | 0.0212 | 400 | −3.8% |
| xgb_isotonic | 0.5808 | 0.0172 | 340 | −9.8% |
| rf_isotonic | **0.5762** | **0.0093** | 279 | **−18.5%** |

Random Forest with isotonic calibration wins on Brier, RPS *and* ECE — and
loses 18.5% of turnover on held-out seasons. Uncalibrated XGBoost is worse on
three of four metrics and is the only variant that returns a profit.

The explanation: aggregate calibration is measured across *all* predictions,
but betting only samples the tail where the model disagrees with the market.
Isotonic regression is a monotone step function fitted on sparse tail bins, so
it flattens precisely the disagreements the strategy trades on — 279 qualifying
bets instead of 877. A model can be well calibrated on average and useless
where you act on it.

Consequence: **the backtest selected the served model, not the scoring rule.**
If you only read `reports/model_metrics.csv`, you would ship the wrong model.

## 3. Best-of-book execution may not be achievable

The headline uses the maximum price across bookmakers. That assumes accounts at
every book in the sample, catching the best price before it moves, and — the
binding constraint — not being restricted. Bettors who consistently beat the
closing line get stake-limited or closed, often within months.

Treat +7.2% as an upper bound and the single-bookmaker +3.2% as the realistic
ceiling. Both are reported for that reason.

## 4. The risk profile is harsher than the return suggests

Maximum drawdown is **−47%**, and 2021/22 lost 11.2% across a full season. Six
of twelve seasons lost money. The simulation assumes a bettor who keeps staking
through all of that without adjusting, which is not how people behave.

Quarter-Kelly and a 10% daily exposure cap are already deliberate concessions:
full Kelly on probabilities estimated with error approaches ruin.

## 5. Feature limitations

- **xG is a proxy.** A Poisson model over shots, shots on target and corners —
  not shot-level expected goals from tracking data.
- **Rolling form is not venue-split.** A team's last five matches alternate
  home and away, so the feature blends two different contexts.
- **No player availability.** Injuries, suspensions and rotation are unmodelled,
  and are probably the largest single reason the market out-discriminates this
  model.
- **457 of 9,380 fixtures are excluded** as burn-in, where Elo and rolling
  windows have no history. They are dropped, not imputed — which slightly
  flatters feature quality relative to a live system facing a promoted club's
  first fixtures.

## 6. The calibration claim is directional, not significant

The model's ECE of 0.0093 against the market's 0.0105 is a 0.0012 gap with no
confidence interval attached. It should not carry weight on its own. The
load-bearing evidence that the probabilities are useful is the backtest, not
that margin.

## 7. Stale features are served, and labelled

The API resolves team ratings from each side's most recent fixture. For a
relegated club those ratings are more than a year old — ask about Burnley and
the response carries `as_of: 2024-05-19`. The value is surfaced rather than
hidden, but it is still stale, and there is no automatic decay or refusal.

---

## What would move these

In rough order of expected value:

1. **Closing-line value tracking** — whether the model beats the price the
   market settles at. Reaches significance in hundreds of bets where ROI needs
   thousands, so it is the leading indicator of a real edge.
2. **Player availability data** — the largest unmodelled signal.
3. **Real shot-level xG** — replaces the proxy in §5.
4. **Venue-split rolling form** — removes the alternation bias.
5. **Re-fitting the selection band every season** and reporting only
   out-of-sample results, so §1 stops being a caveat and becomes the method.
