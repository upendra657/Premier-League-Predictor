"""Decision and financial backtesting engine.

Turns calibrated probabilities into staking decisions and measures what those
decisions would have earned against real historical bookmaker prices.

The economics
-------------
A bookmaker's quoted probabilities sum to more than one; the excess is the
overround, and it is the hurdle every strategy must clear before it earns a
penny. On this dataset a single-book line carries roughly 5.3% overround,
while taking the best price available across books cuts it to about 0.35%.
Execution quality therefore matters as much as model quality, and the engine
reports both so the distinction is explicit rather than assumed away.

Staking uses the Kelly criterion, which maximises the long-run growth rate of
a bankroll. Full Kelly is optimal only if the probabilities are exactly right;
because they never are, the engine deploys a fraction of the Kelly stake,
which sacrifices a little growth for a large reduction in variance and
drawdown risk.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src import config
from src.config import BacktestConfig, StrategyParams

logger = logging.getLogger(__name__)

OUTCOMES: tuple[str, ...] = ("home", "draw", "away")


@dataclass
class BacktestResult:
    """Bet-level ledger plus headline performance statistics."""

    bets: pd.DataFrame
    summary: dict[str, float]
    equity_curve: pd.DataFrame

    def __repr__(self) -> str:  # pragma: no cover - display helper
        rows = "\n".join(f"  {k:<24} {v}" for k, v in self.summary.items())
        return f"BacktestResult(\n{rows}\n)"


# --------------------------------------------------------------------------- #
# Candidate construction
# --------------------------------------------------------------------------- #


def build_candidates(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    odds_source: str = "best_odds",
) -> pd.DataFrame:
    """Explode one row per fixture into three candidate bets.

    Every match offers three mutually exclusive selections; the engine
    evaluates all of them and lets the edge filter decide which, if any,
    are worth backing.

    Parameters
    ----------
    predictions
        Out-of-sample model probabilities, one row per fixture.
    market
        Feature frame carrying odds columns and the realised outcome.
    odds_source
        ``"odds"`` for the single-book average line, ``"best_odds"`` for the
        best price available across books (line shopping).
    """
    odds_columns = [f"{odds_source}_{outcome}" for outcome in OUTCOMES]
    market_columns = ["match_id", "Season", "MatchDate", "HomeTeam", "AwayTeam", "target"]

    merged = predictions.merge(
        market[market_columns + odds_columns].drop_duplicates("match_id"),
        on=[c for c in market_columns if c in predictions.columns],
        how="inner",
        validate="many_to_one",
    ).dropna(subset=odds_columns)

    frames = []
    for index, outcome in enumerate(OUTCOMES):
        frame = merged[["match_id", "Season", "MatchDate", "HomeTeam", "AwayTeam"]].copy()
        frame["outcome"] = outcome
        frame["model_prob"] = merged[f"prob_{outcome}"].to_numpy()
        frame["odds"] = merged[f"{odds_source}_{outcome}"].to_numpy()
        frame["won"] = (merged["target"].to_numpy() == index).astype(int)
        frames.append(frame)

    candidates = pd.concat(frames, ignore_index=True)

    # De-vig the quoted prices so the comparison is model-vs-market rather
    # than model-vs-market-plus-margin.
    implied = 1.0 / candidates["odds"]
    booksum = implied.groupby(candidates["match_id"]).transform("sum")
    candidates["implied_prob"] = implied
    candidates["fair_prob"] = implied / booksum
    candidates["overround"] = booksum - 1.0

    candidates["edge"] = candidates["model_prob"] - candidates["fair_prob"]
    candidates["relative_edge"] = candidates["model_prob"] / candidates["fair_prob"] - 1.0
    # Expected profit per unit staked at the quoted (vig-inclusive) price.
    candidates["expected_value"] = candidates["model_prob"] * candidates["odds"] - 1.0
    candidates["unit_return"] = np.where(
        candidates["won"] == 1, candidates["odds"] - 1.0, -1.0
    )
    return candidates.sort_values(["MatchDate", "match_id"]).reset_index(drop=True)


def kelly_fraction(model_prob: np.ndarray, odds: np.ndarray) -> np.ndarray:
    """Full Kelly stake as a fraction of bankroll.

    For a binary bet at decimal odds ``o`` with win probability ``p``, the
    growth-optimal stake is ``(p*o - 1) / (o - 1)``. Negative values mean the
    bet is unprofitable and are clipped to zero.
    """
    net_odds = odds - 1.0
    with np.errstate(divide="ignore", invalid="ignore"):
        fraction = (model_prob * odds - 1.0) / net_odds
    return np.clip(np.nan_to_num(fraction, nan=0.0, posinf=0.0, neginf=0.0), 0.0, 1.0)


def select_bets(
    candidates: pd.DataFrame,
    cfg: BacktestConfig,
    strategy: StrategyParams | None = None,
) -> pd.DataFrame:
    """Apply the selection band and size each position. Fully vectorised."""
    strategy = strategy or cfg.strategy
    selected = candidates.loc[
        (candidates["edge"] >= strategy.min_edge)
        & (candidates["relative_edge"] >= strategy.min_relative_edge)
        & (candidates["relative_edge"] <= strategy.max_relative_edge)
        & (candidates["expected_value"] > 0.0)
        & (candidates["odds"] >= strategy.min_odds)
        & (candidates["odds"] <= strategy.max_odds)
    ].copy()

    selected["kelly_full"] = kelly_fraction(
        selected["model_prob"].to_numpy(), selected["odds"].to_numpy()
    )
    selected["stake_fraction"] = np.minimum(
        selected["kelly_full"] * cfg.kelly_fraction, cfg.max_stake_fraction
    )
    logger.info(
        "Selected %d bets from %d candidates (%.1f%% selection rate).",
        len(selected),
        len(candidates),
        100.0 * len(selected) / max(len(candidates), 1),
    )
    return selected.reset_index(drop=True)


def calibrate_strategy(
    candidates: pd.DataFrame,
    cfg: BacktestConfig,
    min_bets: int = 300,
) -> tuple[StrategyParams, pd.DataFrame]:
    """Grid-search the selection band on the calibration seasons only.

    Returning the chosen band *and* the full grid keeps the exercise auditable:
    a band that is profitable only at one knife-edge setting is overfitting,
    and that is visible in the grid rather than hidden by it.
    """
    window = candidates.loc[candidates["Season"].isin(cfg.calibration_seasons)].copy()
    window["unit_return"] = np.where(
        window["won"] == 1, window["odds"] - 1.0, -1.0
    )

    rows = []
    for min_rel in (0.05, 0.10, 0.15, 0.20):
        for max_rel in (0.30, 0.40, 0.50, 0.75):
            if max_rel <= min_rel:
                continue
            for max_odds in (4.0, 5.0, 6.5):
                mask = (
                    (window["relative_edge"] >= min_rel)
                    & (window["relative_edge"] <= max_rel)
                    & (window["odds"] >= cfg.strategy.min_odds)
                    & (window["odds"] <= max_odds)
                    & (window["edge"] >= cfg.strategy.min_edge)
                )
                subset = window.loc[mask]
                if len(subset) < min_bets:
                    continue
                rows.append(
                    {
                        "min_relative_edge": min_rel,
                        "max_relative_edge": max_rel,
                        "max_odds": max_odds,
                        "n_bets": len(subset),
                        "roi": float(subset["unit_return"].mean()),
                    }
                )

    grid = pd.DataFrame(rows).sort_values("roi", ascending=False).reset_index(drop=True)
    if grid.empty:
        logger.warning("Strategy grid empty; falling back to configured defaults.")
        return cfg.strategy, grid

    best = grid.iloc[0]
    chosen = StrategyParams(
        min_edge=cfg.strategy.min_edge,
        min_relative_edge=float(best["min_relative_edge"]),
        max_relative_edge=float(best["max_relative_edge"]),
        min_odds=cfg.strategy.min_odds,
        max_odds=float(best["max_odds"]),
    )
    logger.info(
        "Calibrated band on %d seasons: rel_edge in [%.2f, %.2f], odds <= %.1f "
        "(in-sample ROI %.2f%% on %d bets).",
        len(cfg.calibration_seasons),
        chosen.min_relative_edge,
        chosen.max_relative_edge,
        chosen.max_odds,
        best["roi"] * 100,
        int(best["n_bets"]),
    )
    return chosen, grid


# --------------------------------------------------------------------------- #
# Bankroll simulation
# --------------------------------------------------------------------------- #


def simulate_kelly(bets: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Compound a bankroll through the bet ledger, settling day by day.

    Bets placed on the same matchday are sized against the bankroll as it
    stood that morning -- a bettor cannot recycle winnings from a 3pm kick-off
    into another 3pm kick-off. Within a day the stake computation is fully
    vectorised; only the day-to-day carry is sequential.
    """
    bets = bets.sort_values("MatchDate").reset_index(drop=True)
    bankroll = cfg.starting_bankroll

    stakes = np.zeros(len(bets))
    profits = np.zeros(len(bets))
    bankroll_before = np.zeros(len(bets))

    for _, day_index in bets.groupby("MatchDate", sort=True).groups.items():
        positions = np.asarray(day_index)
        day_fraction = bets.loc[positions, "stake_fraction"].to_numpy()
        day_odds = bets.loc[positions, "odds"].to_numpy()
        day_won = bets.loc[positions, "won"].to_numpy()

        day_stakes = day_fraction * bankroll
        # Kelly is derived for one bet at a time. Premier League fixtures
        # cluster on Saturdays, so without a joint exposure cap a dozen
        # "small" positions compound into a single large one and the
        # resulting variance drag destroys the bankroll.
        exposure_cap = bankroll * cfg.max_daily_exposure
        exposure = day_stakes.sum()
        if exposure > exposure_cap:
            day_stakes *= exposure_cap / exposure

        day_profit = np.where(day_won == 1, day_stakes * (day_odds - 1.0), -day_stakes)

        bankroll_before[positions] = bankroll
        stakes[positions] = day_stakes
        profits[positions] = day_profit
        bankroll += day_profit.sum()

    ledger = bets.copy()
    ledger["bankroll_before"] = bankroll_before
    ledger["stake"] = stakes
    ledger["profit"] = profits
    ledger["bankroll_after"] = cfg.starting_bankroll + ledger["profit"].cumsum()
    return ledger


def simulate_flat(bets: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """Benchmark strategy: identical selections, constant stake size."""
    ledger = bets.sort_values("MatchDate").reset_index(drop=True).copy()
    stake = cfg.starting_bankroll * cfg.flat_stake_fraction
    ledger["stake"] = stake
    ledger["profit"] = np.where(
        ledger["won"] == 1, stake * (ledger["odds"] - 1.0), -stake
    )
    ledger["bankroll_after"] = cfg.starting_bankroll + ledger["profit"].cumsum()
    return ledger


# --------------------------------------------------------------------------- #
# Performance analytics
# --------------------------------------------------------------------------- #


def bootstrap_roi_interval(
    unit_returns: np.ndarray,
    n_samples: int,
    seed: int,
    cluster: np.ndarray | None = None,
) -> dict[str, float]:
    """Bootstrap a confidence interval for return on turnover.

    Betting outcomes are extremely heavy-tailed, so a normal-approximation
    standard error understates uncertainty. When ``cluster`` is supplied the
    resampling is done over clusters (seasons) rather than individual bets,
    which respects the fact that a whole season can be good or bad together.
    """
    rng = np.random.default_rng(seed)

    if cluster is not None:
        groups = pd.Series(unit_returns).groupby(cluster).mean().to_numpy()
        draws = rng.choice(groups, size=(n_samples, len(groups)), replace=True).mean(axis=1)
    else:
        draws = rng.choice(
            unit_returns, size=(n_samples, len(unit_returns)), replace=True
        ).mean(axis=1)

    return {
        "roi_ci_low": round(float(np.percentile(draws, 2.5)), 4),
        "roi_ci_high": round(float(np.percentile(draws, 97.5)), 4),
        "p_roi_not_positive": round(float(np.mean(draws <= 0.0)), 4),
    }


def max_drawdown(equity: pd.Series) -> float:
    """Largest peak-to-trough decline in the bankroll, as a fraction."""
    running_peak = equity.cummax()
    return float(((equity - running_peak) / running_peak).min())


def summarise(ledger: pd.DataFrame, cfg: BacktestConfig, label: str) -> dict[str, float]:
    """Compute headline risk and return statistics for a bet ledger."""
    total_staked = float(ledger["stake"].sum())
    total_profit = float(ledger["profit"].sum())
    final_bankroll = float(cfg.starting_bankroll + total_profit)

    per_bet_return = ledger["profit"] / ledger["stake"].replace(0.0, np.nan)
    n_seasons = max(ledger["Season"].nunique(), 1)

    return {
        "strategy": label,
        "n_bets": int(len(ledger)),
        "n_seasons": int(n_seasons),
        "hit_rate": round(float(ledger["won"].mean()), 4),
        "avg_odds": round(float(ledger["odds"].mean()), 3),
        "avg_edge": round(float(ledger["edge"].mean()), 4),
        "total_staked": round(total_staked, 2),
        "total_profit": round(total_profit, 2),
        "roi_on_turnover": round(total_profit / total_staked, 4) if total_staked else 0.0,
        "final_bankroll": round(final_bankroll, 2),
        "bankroll_growth": round(final_bankroll / cfg.starting_bankroll - 1.0, 4),
        "max_drawdown": round(max_drawdown(ledger["bankroll_after"]), 4),
        "sharpe_per_bet": round(
            float(per_bet_return.mean() / per_bet_return.std()), 4
        )
        if per_bet_return.std() > 0
        else 0.0,
    }


def equity_by_season(ledger: pd.DataFrame) -> pd.DataFrame:
    """Season-level profit and loss attribution."""
    grouped = ledger.groupby("Season").agg(
        n_bets=("won", "size"),
        hit_rate=("won", "mean"),
        staked=("stake", "sum"),
        profit=("profit", "sum"),
    )
    grouped["roi"] = grouped["profit"] / grouped["staked"]
    grouped["cumulative_profit"] = grouped["profit"].cumsum()
    return grouped.round(4).reset_index()


def run_backtest(
    predictions: pd.DataFrame,
    market: pd.DataFrame,
    odds_source: str = "best_odds",
    cfg: BacktestConfig | None = None,
    strategy: StrategyParams | None = None,
    seasons: tuple[str, ...] | None = None,
    label: str = "",
) -> BacktestResult:
    """End-to-end: build candidates, filter to EV+ bets, simulate, report."""
    cfg = cfg or config.BACKTEST_CONFIG
    candidates = build_candidates(predictions, market, odds_source=odds_source)
    if seasons is not None:
        candidates = candidates.loc[candidates["Season"].isin(seasons)]

    bets = select_bets(candidates, cfg, strategy)
    if bets.empty:
        empty = pd.DataFrame()
        return BacktestResult(empty, {"strategy": label or "kelly", "n_bets": 0}, empty)

    kelly_ledger = simulate_kelly(bets, cfg)
    flat_ledger = simulate_flat(bets, cfg)

    summary = summarise(kelly_ledger, cfg, label or f"fractional_kelly@{odds_source}")
    # Flat staking isolates selection quality from sizing: with a constant
    # stake, ROI on turnover is just the mean return per unit risked.
    summary["flat_stake_roi"] = summarise(flat_ledger, cfg, "flat")["roi_on_turnover"]
    summary["market_overround"] = round(float(candidates["overround"].mean()), 4)
    summary.update(
        bootstrap_roi_interval(
            bets["unit_return"].to_numpy(),
            cfg.bootstrap_samples,
            cfg.random_state,
            cluster=bets["Season"].to_numpy(),
        )
    )

    return BacktestResult(
        bets=kelly_ledger,
        summary=summary,
        equity_curve=equity_by_season(kelly_ledger),
    )


def main() -> pd.DataFrame:
    """Calibrate the selection band, then report strictly held-out performance.

    The band is fitted on ``cfg.calibration_seasons`` and evaluated on every
    later season. Reporting the held-out number is the whole point: a betting
    strategy tuned and scored on the same data will always look profitable.
    """
    cfg = config.BACKTEST_CONFIG
    predictions = pd.read_parquet(config.PROCESSED_DIR / "oos_predictions.parquet")
    market = pd.read_parquet(config.FEATURE_STORE_FILE)

    all_seasons = sorted(predictions["Season"].unique())
    holdout = tuple(s for s in all_seasons if s not in cfg.calibration_seasons)

    rows = []
    for variant, group in predictions.groupby("variant"):
        for odds_source in ("odds", "best_odds"):
            candidates = build_candidates(group, market, odds_source=odds_source)
            strategy, _grid = calibrate_strategy(candidates, cfg)

            for period_label, seasons in (
                ("calibration", cfg.calibration_seasons),
                ("holdout", holdout),
            ):
                result = run_backtest(
                    group,
                    market,
                    odds_source=odds_source,
                    cfg=cfg,
                    strategy=strategy,
                    seasons=seasons,
                    label=f"{variant}|{odds_source}|{period_label}",
                )
                if not result.summary.get("n_bets"):
                    continue
                rows.append(
                    {
                        "variant": variant,
                        "execution": odds_source,
                        "period": period_label,
                        **result.summary,
                    }
                )

    comparison = pd.DataFrame(rows)
    comparison.to_csv(config.REPORT_DIR / "backtest_comparison.csv", index=False)

    holdout_view = (
        comparison.loc[comparison["period"] == "holdout"]
        .sort_values("flat_stake_roi", ascending=False)
        .reset_index(drop=True)
    )
    print("\nHeld-out backtest performance (band fitted on 2013/14-2018/19)\n")
    print(
        holdout_view[
            [
                "variant", "execution", "n_bets", "hit_rate", "avg_odds",
                "flat_stake_roi", "roi_ci_low", "roi_ci_high", "p_roi_not_positive",
                "bankroll_growth", "max_drawdown",
            ]
        ].to_string(index=False)
    )
    return comparison


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
