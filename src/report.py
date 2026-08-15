"""Consolidated results reporting.

Produces the numbers quoted in the README: calibration gains, model-versus-
market comparison, and backtest performance under both the pre-registered
selection band and the band fitted on the calibration seasons.
"""

from __future__ import annotations

import json
import logging

import numpy as np
import pandas as pd

from src import config
from src.backtest import (
    build_candidates,
    calibrate_strategy,
    run_backtest,
)

logger = logging.getLogger(__name__)


def calibration_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Quantify what post-hoc calibration bought, per base estimator."""
    indexed = metrics.set_index("model")
    rows = []
    for family, raw, calibrated in (
        ("random_forest", "rf_uncalibrated", "rf_isotonic"),
        ("xgboost", "xgb_uncalibrated", "xgb_isotonic"),
    ):
        if raw not in indexed.index or calibrated not in indexed.index:
            continue
        before, after = indexed.loc[raw], indexed.loc[calibrated]
        rows.append(
            {
                "estimator": family,
                "brier_before": before["brier"],
                "brier_after": after["brier"],
                "brier_delta_pct": round(
                    100 * (after["brier"] - before["brier"]) / before["brier"], 2
                ),
                "ece_before": before["ece"],
                "ece_after": after["ece"],
                "ece_delta_pct": round(
                    100 * (after["ece"] - before["ece"]) / before["ece"], 2
                ),
                "log_loss_before": before["log_loss"],
                "log_loss_after": after["log_loss"],
            }
        )
    return pd.DataFrame(rows)


def reliability_curve(
    predictions: pd.DataFrame, variant: str, n_bins: int = 10
) -> pd.DataFrame:
    """Predicted-versus-observed frequency, pooled one-vs-rest across classes."""
    subset = predictions.loc[predictions["variant"] == variant]
    probabilities = subset[["prob_home", "prob_draw", "prob_away"]].to_numpy()
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(subset)), subset["target"].to_numpy()] = 1.0

    frame = pd.DataFrame(
        {"predicted": probabilities.ravel(), "observed": one_hot.ravel()}
    )
    frame["bin"] = pd.cut(frame["predicted"], np.linspace(0, 1, n_bins + 1))
    grouped = frame.groupby("bin", observed=True).agg(
        n=("observed", "size"),
        mean_predicted=("predicted", "mean"),
        observed_rate=("observed", "mean"),
    )
    grouped["gap"] = grouped["mean_predicted"] - grouped["observed_rate"]
    return grouped.round(4).reset_index()


def edge_profile(candidates: pd.DataFrame) -> pd.DataFrame:
    """Realised return by relative-edge bucket -- the core economic finding."""
    buckets = pd.cut(
        candidates["relative_edge"], [-1.0, 0.0, 0.05, 0.10, 0.20, 0.40, 10.0]
    )
    grouped = candidates.groupby(buckets, observed=True).agg(
        n=("unit_return", "size"),
        hit_rate=("won", "mean"),
        avg_odds=("odds", "mean"),
        roi=("unit_return", "mean"),
    )
    return grouped.round(4).reset_index()


def main() -> dict:
    """Assemble every headline figure and write the results bundle."""
    metrics = pd.read_csv(config.REPORT_DIR / "model_metrics.csv")
    predictions = pd.read_parquet(config.PROCESSED_DIR / "oos_predictions.parquet")
    market = pd.read_parquet(config.FEATURE_STORE_FILE)
    cfg = config.BACKTEST_CONFIG

    best_variant = "xgb_uncalibrated"
    calib = calibration_table(metrics)
    curve = reliability_curve(predictions, best_variant)

    candidates = build_candidates(
        predictions[predictions["variant"] == best_variant], market, "best_odds"
    )
    profile = edge_profile(candidates)

    # (a) Pre-registered band applied to the whole evaluation period.
    full = run_backtest(
        predictions[predictions["variant"] == best_variant],
        market,
        odds_source="best_odds",
        label="preregistered_band_full_period",
    )

    # (b) Band fitted on the calibration seasons, scored on held-out seasons.
    strategy, grid = calibrate_strategy(candidates, cfg)
    holdout_seasons = tuple(
        s for s in sorted(predictions["Season"].unique())
        if s not in cfg.calibration_seasons
    )
    holdout = run_backtest(
        predictions[predictions["variant"] == best_variant],
        market,
        odds_source="best_odds",
        strategy=strategy,
        seasons=holdout_seasons,
        label="fitted_band_holdout",
    )

    single_book = run_backtest(
        predictions[predictions["variant"] == best_variant],
        market,
        odds_source="odds",
        label="preregistered_band_single_book",
    )

    results = {
        "dataset": {
            "fixtures": int(len(market)),
            "seasons": int(market["Season"].nunique()),
            "modellable": int(market["is_modellable"].sum()),
            "odds_coverage": round(float(market["odds_home"].notna().mean()), 4),
        },
        "walk_forward_metrics": metrics.to_dict(orient="records"),
        "calibration_gain": calib.to_dict(orient="records"),
        "reliability_curve": curve.to_dict(orient="records"),
        "edge_profile": profile.to_dict(orient="records"),
        "backtest": {
            "full_period": full.summary,
            "holdout": holdout.summary,
            "single_book": single_book.summary,
            "fitted_band": {
                "min_relative_edge": strategy.min_relative_edge,
                "max_relative_edge": strategy.max_relative_edge,
                "max_odds": strategy.max_odds,
            },
        },
        "equity_by_season": full.equity_curve.to_dict(orient="records"),
    }

    (config.REPORT_DIR / "results.json").write_text(json.dumps(results, indent=2, default=str))
    full.equity_curve.to_csv(config.REPORT_DIR / "equity_by_season.csv", index=False)
    curve.to_csv(config.REPORT_DIR / "reliability_curve.csv", index=False)
    profile.to_csv(config.REPORT_DIR / "edge_profile.csv", index=False)
    grid.to_csv(config.REPORT_DIR / "strategy_grid.csv", index=False)

    print("\n== Calibration gain ==")
    print(calib.to_string(index=False))
    print("\n== Realised return by relative-edge bucket ==")
    print(profile.to_string(index=False))
    print("\n== Backtest ==")
    for key in ("full_period", "holdout", "single_book"):
        print(f"\n[{key}]")
        for name, value in results["backtest"][key].items():
            print(f"  {name:<24} {value}")
    print("\n== Season equity ==")
    print(full.equity_curve.to_string(index=False))
    return results


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    main()
