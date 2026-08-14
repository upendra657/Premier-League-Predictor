"""Model training, probability calibration and walk-forward evaluation.

The objective here is *not* accuracy. A decision engine that sizes financial
stakes needs probabilities that mean what they say: when the model says 30%,
the event should happen 30% of the time. Accuracy is blind to that property,
so every metric in this module is a proper scoring rule -- multiclass Brier
score and log loss -- benchmarked against the de-vigged bookmaker line, which
is the strongest publicly available forecast of a football match.

Evaluation is walk-forward by season: for each evaluation season the model
sees only seasons that finished before it. This mirrors deployment and makes
the reported numbers honest.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import TimeSeriesSplit
from xgboost import XGBClassifier

from src import config
from src.config import TrainConfig
from src.features import MODEL_FEATURES

logger = logging.getLogger(__name__)

MARKET_PROB_COLUMNS: list[str] = ["mkt_prob_home", "mkt_prob_draw", "mkt_prob_away"]
PRED_PROB_COLUMNS: list[str] = ["prob_home", "prob_draw", "prob_away"]


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def multiclass_brier_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Mean squared error between the probability vector and the outcome.

    Defined as ``mean_i sum_k (p_ik - y_ik)^2`` where ``y`` is one-hot. Lower
    is better; the theoretical range for three classes is [0, 2]. Always
    predicting the base rate scores roughly 0.62 on Premier League outcomes.
    """
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    return float(np.mean(np.sum((probabilities - one_hot) ** 2, axis=1)))


def ranked_probability_score(y_true: np.ndarray, probabilities: np.ndarray) -> float:
    """Ordinal-aware scoring rule for Home/Draw/Away outcomes.

    Match results are ordered (a home win is 'further' from an away win than
    from a draw). RPS penalises probability mass placed far from the truth on
    that ordering, and is the standard metric in football forecasting papers.
    """
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    cumulative_pred = np.cumsum(probabilities, axis=1)
    cumulative_true = np.cumsum(one_hot, axis=1)
    n_classes = probabilities.shape[1]
    return float(
        np.mean(np.sum((cumulative_pred - cumulative_true) ** 2, axis=1) / (n_classes - 1))
    )


def expected_calibration_error(
    y_true: np.ndarray, probabilities: np.ndarray, n_bins: int = 10
) -> float:
    """Average gap between predicted confidence and observed frequency.

    Computed one-vs-rest across all three classes and pooled, which surfaces
    miscalibration on the draw class that a top-label-only metric would hide.
    """
    one_hot = np.zeros_like(probabilities)
    one_hot[np.arange(len(y_true)), y_true] = 1.0
    flat_pred = probabilities.ravel()
    flat_true = one_hot.ravel()

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_index = np.clip(np.digitize(flat_pred, bins) - 1, 0, n_bins - 1)

    total_error = 0.0
    for b in range(n_bins):
        mask = bin_index == b
        if not mask.any():
            continue
        weight = mask.mean()
        total_error += weight * abs(flat_pred[mask].mean() - flat_true[mask].mean())
    return float(total_error)


def score_predictions(
    y_true: np.ndarray, probabilities: np.ndarray, label: str
) -> dict[str, float]:
    """Bundle every evaluation metric for one set of predictions."""
    return {
        "model": label,
        "n": int(len(y_true)),
        "brier": round(multiclass_brier_score(y_true, probabilities), 5),
        "log_loss": round(float(log_loss(y_true, probabilities, labels=[0, 1, 2])), 5),
        "rps": round(ranked_probability_score(y_true, probabilities), 5),
        "ece": round(expected_calibration_error(y_true, probabilities), 5),
        "accuracy": round(float(np.mean(probabilities.argmax(axis=1) == y_true)), 5),
    }


# --------------------------------------------------------------------------- #
# Estimators
# --------------------------------------------------------------------------- #


def build_estimator(name: str, cfg: TrainConfig):
    """Instantiate a base classifier by name."""
    if name == "xgboost":
        return XGBClassifier(random_state=cfg.random_state, **cfg.xgb_params)
    if name == "random_forest":
        return RandomForestClassifier(random_state=cfg.random_state, **cfg.rf_params)
    raise ValueError(f"Unknown estimator: {name!r}")


def build_calibrated_estimator(name: str, cfg: TrainConfig, method: str):
    """Wrap a base estimator in a time-series-aware probability calibrator.

    ``CalibratedClassifierCV`` refits the base model on each fold and learns
    the mapping from raw scores to calibrated probabilities on the held-out
    part. Passing a ``TimeSeriesSplit`` keeps folds chronological, so the
    calibrator is never fitted on data that post-dates its validation slice.
    """
    return CalibratedClassifierCV(
        estimator=build_estimator(name, cfg),
        method=method,
        cv=TimeSeriesSplit(n_splits=cfg.cv_splits),
        ensemble=True,
        n_jobs=1,
    )


def align_probabilities(estimator, raw_probabilities: np.ndarray) -> np.ndarray:
    """Reorder estimator output into the canonical [H, D, A] ordering."""
    order = [list(estimator.classes_).index(i) for i in range(len(config.CLASS_LABELS))]
    return raw_probabilities[:, order]


# --------------------------------------------------------------------------- #
# Walk-forward evaluation
# --------------------------------------------------------------------------- #


@dataclass
class WalkForwardResult:
    """Container for out-of-sample predictions and per-variant metrics."""

    predictions: pd.DataFrame
    metrics: pd.DataFrame


def walk_forward_evaluate(
    features: pd.DataFrame,
    variants: dict[str, tuple[str, str | None]],
    cfg: TrainConfig | None = None,
) -> WalkForwardResult:
    """Evaluate every model variant season by season on strictly unseen data.

    Parameters
    ----------
    features
        Feature frame from :func:`src.features.build_features`.
    variants
        Mapping of ``label -> (estimator_name, calibration_method)``. A
        calibration method of ``None`` trains the raw estimator.

    Returns
    -------
    WalkForwardResult
        Row-level out-of-sample predictions plus aggregate metrics.
    """
    cfg = cfg or config.TRAIN_CONFIG
    modellable = features.loc[features["is_modellable"]].copy()
    modellable = modellable.sort_values("MatchDate").reset_index(drop=True)

    seasons = sorted(modellable["Season"].unique())
    eval_seasons = [s for s in seasons if s >= cfg.eval_start_season]
    logger.info(
        "Walk-forward over %d evaluation seasons (%s to %s).",
        len(eval_seasons),
        eval_seasons[0],
        eval_seasons[-1],
    )

    collected: list[pd.DataFrame] = []

    for season in eval_seasons:
        train_mask = modellable["Season"] < season
        test_mask = modellable["Season"] == season
        x_train = modellable.loc[train_mask, list(MODEL_FEATURES)]
        y_train = modellable.loc[train_mask, "target"].to_numpy()
        x_test = modellable.loc[test_mask, list(MODEL_FEATURES)]

        season_frame = modellable.loc[
            test_mask,
            ["match_id", "Season", "MatchDate", "HomeTeam", "AwayTeam", "target"],
        ].copy()

        for label, (estimator_name, method) in variants.items():
            estimator = (
                build_estimator(estimator_name, cfg)
                if method is None
                else build_calibrated_estimator(estimator_name, cfg, method)
            )
            estimator.fit(x_train, y_train)
            probabilities = align_probabilities(estimator, estimator.predict_proba(x_test))

            variant_frame = season_frame.copy()
            variant_frame["variant"] = label
            variant_frame[PRED_PROB_COLUMNS] = probabilities
            collected.append(variant_frame)

        logger.info("  %s: trained on %d rows, scored %d.", season, int(train_mask.sum()), int(test_mask.sum()))

    predictions = pd.concat(collected, ignore_index=True)

    metrics = [
        score_predictions(
            group["target"].to_numpy(),
            group[PRED_PROB_COLUMNS].to_numpy(),
            label,
        )
        for label, group in predictions.groupby("variant")
    ]

    # Market benchmark evaluated on exactly the same fixtures.
    benchmark_ids = predictions["match_id"].unique()
    market = modellable.loc[
        modellable["match_id"].isin(benchmark_ids)
        & modellable[MARKET_PROB_COLUMNS].notna().all(axis=1)
    ]
    # De-vigging leaves a float-precision residue; renormalise so the
    # benchmark is scored on vectors that sum to exactly one.
    market_probs = market[MARKET_PROB_COLUMNS].to_numpy()
    market_probs = market_probs / market_probs.sum(axis=1, keepdims=True)
    metrics.append(
        score_predictions(market["target"].to_numpy(), market_probs, "market_baseline")
    )

    metrics_frame = pd.DataFrame(metrics).sort_values("brier").reset_index(drop=True)
    return WalkForwardResult(predictions=predictions, metrics=metrics_frame)


# --------------------------------------------------------------------------- #
# Final artifact
# --------------------------------------------------------------------------- #


def fit_production_model(
    features: pd.DataFrame,
    estimator_name: str,
    method: str | None,
    cfg: TrainConfig | None = None,
):
    """Refit the deployed configuration on the full modellable history.

    ``method=None`` ships the base estimator directly, which is the correct
    choice when the estimator already trains against a proper scoring rule.
    """
    cfg = cfg or config.TRAIN_CONFIG
    modellable = features.loc[features["is_modellable"]].sort_values("MatchDate")
    estimator = (
        build_estimator(estimator_name, cfg)
        if method is None
        else build_calibrated_estimator(estimator_name, cfg, method)
    )
    estimator.fit(modellable[list(MODEL_FEATURES)], modellable["target"])
    logger.info(
        "Production model (%s, calibration=%s) fitted on %d fixtures.",
        estimator_name,
        method or "native",
        len(modellable),
    )
    return estimator


def persist_artifacts(
    estimator,
    shot_model,
    metrics: pd.DataFrame,
    chosen: dict[str, str],
) -> None:
    """Write the model bundle and its metadata card to disk."""
    import joblib

    joblib.dump(
        {"model": estimator, "features": list(MODEL_FEATURES), "classes": config.CLASS_LABELS},
        config.MODEL_FILE,
    )
    joblib.dump(shot_model, config.XG_MODEL_FILE)

    metadata = {
        "estimator": chosen["estimator"],
        "calibration": chosen["method"],
        "features": list(MODEL_FEATURES),
        "classes": list(config.CLASS_LABELS),
        "walk_forward_metrics": metrics.to_dict(orient="records"),
    }
    config.METADATA_FILE.write_text(json.dumps(metadata, indent=2))
    logger.info("Artifacts written to %s.", config.ARTIFACT_DIR)


DEFAULT_VARIANTS: dict[str, tuple[str, str | None]] = {
    "xgb_uncalibrated": ("xgboost", None),
    "xgb_isotonic": ("xgboost", "isotonic"),
    "xgb_sigmoid": ("xgboost", "sigmoid"),
    "rf_uncalibrated": ("random_forest", None),
    "rf_isotonic": ("random_forest", "isotonic"),
}


def main() -> WalkForwardResult:
    """Run the full training and evaluation pipeline end to end."""
    from src.data import build_dataset
    from src.features import build_features

    features, shot_model = build_features(build_dataset())
    features.to_parquet(config.FEATURE_STORE_FILE, index=False)

    result = walk_forward_evaluate(features, DEFAULT_VARIANTS)
    result.predictions.to_parquet(config.PROCESSED_DIR / "oos_predictions.parquet", index=False)
    result.metrics.to_csv(config.REPORT_DIR / "model_metrics.csv", index=False)

    print("\nWalk-forward out-of-sample performance\n")
    print(result.metrics.to_string(index=False))

    cfg = config.TRAIN_CONFIG
    production = fit_production_model(
        features, cfg.production_estimator, cfg.production_calibration
    )
    persist_artifacts(
        production,
        shot_model,
        result.metrics,
        {
            "estimator": cfg.production_estimator,
            "method": cfg.production_calibration or "native (mlogloss objective)",
        },
    )
    return result


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    main()
