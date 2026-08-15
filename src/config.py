"""Central configuration for the Premier League Decision Engine.

All tunable constants live here so that experiments are reproducible and
no magic numbers leak into the pipeline modules.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

# --------------------------------------------------------------------------- #
# Paths
# --------------------------------------------------------------------------- #

ROOT_DIR: Path = Path(__file__).resolve().parents[1]
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
PROCESSED_DIR: Path = DATA_DIR / "processed"
ARTIFACT_DIR: Path = ROOT_DIR / "artifacts"
REPORT_DIR: Path = ROOT_DIR / "reports"

for _directory in (RAW_DIR, PROCESSED_DIR, ARTIFACT_DIR, REPORT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

MATCH_SPINE_FILE: Path = RAW_DIR / "epl_final.csv"
MARKET_ODDS_FILE: Path = RAW_DIR / "market_odds.csv"
FEATURE_STORE_FILE: Path = PROCESSED_DIR / "feature_store.parquet"

MODEL_FILE: Path = ARTIFACT_DIR / "calibrated_model.joblib"
METADATA_FILE: Path = ARTIFACT_DIR / "model_metadata.json"
XG_MODEL_FILE: Path = ARTIFACT_DIR / "shot_quality_model.joblib"
#: Per-team current form, shipped with the model so the inference image
#: does not need the feature store (which is data, not an artifact).
TEAM_SNAPSHOT_FILE: Path = ARTIFACT_DIR / "team_snapshot.json"

# --------------------------------------------------------------------------- #
# Data sources
# --------------------------------------------------------------------------- #

#: Public mirror of football-data.co.uk carrying 1X2 closing odds for E0.
MARKET_ODDS_URL: str = (
    "https://raw.githubusercontent.com/xgabora/"
    "Club-Football-Match-Data-2000-2025/main/data/Matches.csv"
)
DIVISION_CODE: str = "E0"

# --------------------------------------------------------------------------- #
# Target encoding
# --------------------------------------------------------------------------- #

#: Canonical class ordering. Every probability vector in the codebase is
#: ordered [Home win, Draw, Away win] -- never rely on estimator class order.
CLASS_LABELS: tuple[str, ...] = ("H", "D", "A")
CLASS_TO_INDEX: dict[str, int] = {label: i for i, label in enumerate(CLASS_LABELS)}

# --------------------------------------------------------------------------- #
# Feature engineering hyper-parameters
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class EloConfig:
    """Parameters governing the dynamic Elo rating system."""

    initial_rating: float = 1500.0
    #: Base K-factor before goal-difference scaling.
    k_factor: float = 20.0
    #: Points added to the home side's rating when computing win expectancy.
    home_advantage: float = 65.0
    #: Divisor in the logistic expectancy curve (400 => 10x odds per 400 pts).
    scale: float = 400.0
    #: Fraction of a team's rating pulled back to the league mean between
    #: seasons, modelling squad turnover and promotion/relegation churn.
    season_regression: float = 0.25


@dataclass(frozen=True)
class RollingConfig:
    """Parameters for time-decayed rolling form windows."""

    window: int = 5
    #: Exponential decay half-life in matches. A half-life of 2.5 means a
    #: fixture five games ago carries 25% of the weight of the last game.
    half_life: float = 2.5
    #: Matches a team must have played before its rolling features are trusted.
    min_periods: int = 3


@dataclass(frozen=True)
class FeatureConfig:
    """Top-level feature engineering configuration."""

    elo: EloConfig = field(default_factory=EloConfig)
    rolling: RollingConfig = field(default_factory=RollingConfig)
    #: Rest days are clipped to this ceiling so that summer breaks do not
    #: dominate the fatigue signal.
    max_rest_days: int = 14


# --------------------------------------------------------------------------- #
# Training configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TrainConfig:
    """Model training and calibration settings."""

    #: Seasons before this are burn-in only: they seed Elo and rolling form
    #: but are never used as training rows.
    burn_in_seasons: int = 1
    #: First season included in the walk-forward evaluation.
    eval_start_season: str = "2013/14"
    #: Number of expanding-window folds used to fit the calibrator.
    cv_splits: int = 5
    calibration_method: str = "isotonic"
    random_state: int = 42

    #: Estimator shipped to production. XGBoost is chosen deliberately over the
    #: marginally lower-Brier calibrated Random Forest: it is the variant whose
    #: betting edge survives out-of-sample, its `mlogloss` objective calibrates
    #: it natively, and its artifact is ~40x smaller than a 5-fold RF ensemble.
    production_estimator: str = "xgboost"
    #: ``None`` ships the raw estimator. XGBoost optimises a strictly proper
    #: scoring rule during training, so post-hoc isotonic regression measurably
    #: worsened its Brier score here (0.5792 -> 0.5808) by refitting on folds.
    production_calibration: str | None = None

    xgb_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 400,
            "learning_rate": 0.03,
            "max_depth": 4,
            "min_child_weight": 6,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "reg_lambda": 2.0,
            "reg_alpha": 0.5,
            "objective": "multi:softprob",
            "num_class": 3,
            "eval_metric": "mlogloss",
            "tree_method": "hist",
            "n_jobs": -1,
        }
    )
    rf_params: dict = field(
        default_factory=lambda: {
            "n_estimators": 600,
            "max_depth": 12,
            "min_samples_leaf": 25,
            "max_features": "sqrt",
            "class_weight": "balanced_subsample",
            "n_jobs": -1,
        }
    )


# --------------------------------------------------------------------------- #
# Backtesting configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class StrategyParams:
    """Bet selection filters.

    Selection is deliberately a *band*, not a threshold. Empirically the
    model's small disagreements with the market are noise that cannot clear
    the overround, while its largest disagreements are overconfidence on
    longshots. Profitable mispricing lives between those two failure modes.
    """

    #: Minimum absolute edge: model probability minus de-vigged market price.
    min_edge: float = 0.03
    #: Minimum *relative* edge (model / market - 1). Scale-free, so it does
    #: not wave through a 3pp edge on a 5% longshot.
    min_relative_edge: float = 0.10
    #: Upper bound on relative edge -- above this the model is disagreeing
    #: with the market so violently that the model is usually the one at fault.
    max_relative_edge: float = 0.40
    min_odds: float = 1.80
    max_odds: float = 5.00


@dataclass(frozen=True)
class BacktestConfig:
    """Staking and risk parameters for the decision engine."""

    starting_bankroll: float = 1_000.0
    #: Fraction of the full Kelly stake actually deployed. Full Kelly is
    #: variance-optimal but ruinous under model misspecification.
    kelly_fraction: float = 0.25
    #: Hard cap on any single stake as a share of current bankroll.
    max_stake_fraction: float = 0.02
    #: Cap on total capital at risk across simultaneous same-day fixtures.
    max_daily_exposure: float = 0.10
    #: Flat-stake size used for the benchmark strategy, as a share of the
    #: starting bankroll.
    flat_stake_fraction: float = 0.01
    #: Seasons used to fit the selection band. Everything after is held out.
    calibration_seasons: tuple[str, ...] = (
        "2013/14",
        "2014/15",
        "2015/16",
        "2016/17",
        "2017/18",
        "2018/19",
    )
    bootstrap_samples: int = 10_000
    random_state: int = 42

    strategy: StrategyParams = field(default_factory=StrategyParams)


FEATURE_CONFIG = FeatureConfig()
TRAIN_CONFIG = TrainConfig()
BACKTEST_CONFIG = BacktestConfig()
