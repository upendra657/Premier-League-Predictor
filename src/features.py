"""Point-in-time feature engineering for match outcome prediction.

Three families of signal are produced:

1. **Dynamic Elo ratings** -- a sequentially updated strength estimate with a
   margin-of-victory multiplier, home-advantage offset and between-season
   regression to the mean.
2. **Time-decayed rolling expected goals** -- a shot-quality model converts
   each team's shot profile into expected goals, which are then averaged over
   a short window with exponential decay so recent form dominates.
3. **Fatigue** -- rest-day differential between the two sides.

Leakage policy
--------------
Every feature describing a fixture is computed strictly from information
available *before* kick-off. Elo ratings are snapshotted pre-update; rolling
windows are lagged by one match. The shot-quality model is the one component
fitted on data, and it is fitted only on the burn-in seasons that are excluded
from model training and evaluation.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from sklearn.linear_model import PoissonRegressor

from src import config
from src.config import FeatureConfig

logger = logging.getLogger(__name__)

#: Inputs to the shot-quality (expected goals proxy) model.
SHOT_QUALITY_INPUTS: tuple[str, ...] = (
    "shots",
    "shots_on_target",
    "corners",
    "shot_accuracy",
)

#: Columns handed to the classifier. Everything here is strictly pre-match.
MODEL_FEATURES: tuple[str, ...] = (
    "elo_home",
    "elo_away",
    "elo_diff",
    "elo_win_expectancy",
    "xg_home_roll",
    "xg_away_roll",
    "xga_home_roll",
    "xga_away_roll",
    "xg_diff_home",
    "xg_diff_away",
    "xg_matchup_edge",
    "points_home_roll",
    "points_away_roll",
    "form_diff",
    "rest_days_home",
    "rest_days_away",
    "rest_diff",
    "matchweek",
)


# --------------------------------------------------------------------------- #
# Elo
# --------------------------------------------------------------------------- #


def _expected_score(rating_home: float, rating_away: float, cfg) -> float:
    """Logistic win expectancy for the home side, including home advantage."""
    delta = (rating_home + cfg.home_advantage) - rating_away
    return 1.0 / (1.0 + 10.0 ** (-delta / cfg.scale))


def _margin_multiplier(goal_diff: int, rating_delta: float) -> float:
    """Scale the K-factor by margin of victory.

    Uses the FiveThirtyEight formulation: the logarithmic term rewards larger
    margins with diminishing returns, while the denominator damps rating
    inflation when an already-strong favourite wins heavily.
    """
    return np.log(abs(goal_diff) + 1.0) * (2.2 / (abs(rating_delta) * 0.001 + 2.2))


def compute_elo_ratings(frame: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    """Attach pre-match Elo ratings via a single chronological pass.

    Ratings are read *before* the fixture is scored and written back after,
    so the returned columns never contain information from the match itself.
    """
    elo_cfg = cfg.elo
    ratings: dict[str, float] = {}
    current_season: str | None = None

    home_elo = np.empty(len(frame), dtype=float)
    away_elo = np.empty(len(frame), dtype=float)

    columns = frame[
        ["Season", "HomeTeam", "AwayTeam", "FullTimeHomeGoals", "FullTimeAwayGoals"]
    ].to_numpy(dtype=object)

    for i, (season, home, away, home_goals, away_goals) in enumerate(columns):
        if season != current_season:
            # Between seasons, pull every rating partway back to the league
            # mean: squads turn over and promoted sides inherit no history.
            if current_season is not None:
                mean_rating = float(np.mean(list(ratings.values())))
                for team in ratings:
                    ratings[team] += elo_cfg.season_regression * (
                        mean_rating - ratings[team]
                    )
            current_season = season

        rating_home = ratings.setdefault(home, elo_cfg.initial_rating)
        rating_away = ratings.setdefault(away, elo_cfg.initial_rating)
        home_elo[i] = rating_home
        away_elo[i] = rating_away

        expected_home = _expected_score(rating_home, rating_away, elo_cfg)
        goal_diff = int(home_goals) - int(away_goals)
        if goal_diff > 0:
            actual_home = 1.0
        elif goal_diff == 0:
            actual_home = 0.5
        else:
            actual_home = 0.0

        rating_delta = (rating_home + elo_cfg.home_advantage) - rating_away
        multiplier = (
            _margin_multiplier(goal_diff, rating_delta) if goal_diff != 0 else 1.0
        )
        adjustment = elo_cfg.k_factor * multiplier * (actual_home - expected_home)

        ratings[home] = rating_home + adjustment
        ratings[away] = rating_away - adjustment

    frame = frame.copy()
    frame["elo_home"] = home_elo
    frame["elo_away"] = away_elo
    frame["elo_diff"] = home_elo - away_elo
    frame["elo_win_expectancy"] = 1.0 / (
        1.0 + 10.0 ** (-(frame["elo_diff"] + elo_cfg.home_advantage) / elo_cfg.scale)
    )
    return frame


# --------------------------------------------------------------------------- #
# Shot-quality expected goals proxy
# --------------------------------------------------------------------------- #


def to_team_perspective(frame: pd.DataFrame) -> pd.DataFrame:
    """Reshape one row per fixture into two rows, one per participating team.

    Working in team-perspective form lets a single grouped computation serve
    both home and away rolling features.
    """
    shared = ["match_id", "Season", "MatchDate"]

    home = frame[shared].copy()
    home["team"] = frame["HomeTeam"]
    home["opponent"] = frame["AwayTeam"]
    home["is_home"] = 1
    home["goals"] = frame["FullTimeHomeGoals"]
    home["goals_conceded"] = frame["FullTimeAwayGoals"]
    home["shots"] = frame["HomeShots"]
    home["shots_on_target"] = frame["HomeShotsOnTarget"]
    home["corners"] = frame["HomeCorners"]
    home["shots_against"] = frame["AwayShots"]
    home["shots_on_target_against"] = frame["AwayShotsOnTarget"]
    home["corners_against"] = frame["AwayCorners"]

    away = frame[shared].copy()
    away["team"] = frame["AwayTeam"]
    away["opponent"] = frame["HomeTeam"]
    away["is_home"] = 0
    away["goals"] = frame["FullTimeAwayGoals"]
    away["goals_conceded"] = frame["FullTimeHomeGoals"]
    away["shots"] = frame["AwayShots"]
    away["shots_on_target"] = frame["AwayShotsOnTarget"]
    away["corners"] = frame["AwayCorners"]
    away["shots_against"] = frame["HomeShots"]
    away["shots_on_target_against"] = frame["HomeShotsOnTarget"]
    away["corners_against"] = frame["HomeCorners"]

    long_frame = pd.concat([home, away], ignore_index=True)
    long_frame["points"] = np.select(
        [
            long_frame["goals"] > long_frame["goals_conceded"],
            long_frame["goals"] == long_frame["goals_conceded"],
        ],
        [3.0, 1.0],
        default=0.0,
    )
    return long_frame.sort_values(["team", "MatchDate"]).reset_index(drop=True)


def _shot_quality_design(
    shots: pd.Series, on_target: pd.Series, corners: pd.Series
) -> pd.DataFrame:
    """Build the design matrix for the expected goals proxy model.

    Takes the three raw counts explicitly rather than a column prefix, so the
    same routine serves both the attacking and the conceding perspective.
    """
    shots = shots.astype(float)
    on_target = on_target.astype(float)
    accuracy = np.divide(
        on_target.to_numpy(),
        shots.to_numpy(),
        out=np.zeros(len(shots), dtype=float),
        where=shots.to_numpy() > 0,
    )
    return pd.DataFrame(
        {
            "shots": shots.to_numpy(),
            "shots_on_target": on_target.to_numpy(),
            "corners": corners.astype(float).to_numpy(),
            "shot_accuracy": accuracy,
        },
        index=shots.index,
    )


def fit_shot_quality_model(train_frame: pd.DataFrame) -> PoissonRegressor:
    """Fit a Poisson model mapping a shot profile onto expected goals.

    Real xG is derived from per-shot location and situation data, which this
    dataset does not carry. A Poisson regression on shot volume, shots on
    target and corners recovers the same underlying quantity at match level:
    the goals a team 'deserved' from the chances it created, stripped of
    finishing luck.

    Fitted on burn-in seasons only, so no evaluation-period information ever
    reaches the model.
    """
    design = _shot_quality_design(
        train_frame["shots"], train_frame["shots_on_target"], train_frame["corners"]
    )
    model = PoissonRegressor(alpha=1e-3, max_iter=1000)
    model.fit(design, train_frame["goals"].astype(float))
    logger.info(
        "Shot-quality model fitted on %d team-matches; coefficients: %s",
        len(train_frame),
        dict(zip(SHOT_QUALITY_INPUTS, model.coef_.round(4))),
    )
    return model


def apply_shot_quality(
    long_frame: pd.DataFrame, model: PoissonRegressor
) -> pd.DataFrame:
    """Score expected goals for and against each team-match."""
    long_frame = long_frame.copy()
    long_frame["xg"] = model.predict(
        _shot_quality_design(
            long_frame["shots"],
            long_frame["shots_on_target"],
            long_frame["corners"],
        )
    )
    long_frame["xga"] = model.predict(
        _shot_quality_design(
            long_frame["shots_against"],
            long_frame["shots_on_target_against"],
            long_frame["corners_against"],
        )
    )
    return long_frame


# --------------------------------------------------------------------------- #
# Time-decayed rolling windows
# --------------------------------------------------------------------------- #


def decayed_rolling_mean(
    values: pd.Series,
    group: pd.Series,
    window: int,
    half_life: float,
    min_periods: int,
) -> pd.Series:
    """Exponentially weighted mean over the previous ``window`` observations.

    Combines a hard lookback window with exponential time decay: the most
    recent match carries full weight and weight halves every ``half_life``
    matches. Implemented as a fixed linear combination of lagged series, which
    is fully vectorised and avoids the quadratic cost of ``rolling.apply``.

    The result is lagged by one observation, so the value on row *t* uses only
    observations strictly before *t*.
    """
    grouped = values.groupby(group)
    weights = 0.5 ** (np.arange(window) / half_life)

    weighted_total = pd.Series(0.0, index=values.index)
    weight_total = pd.Series(0.0, index=values.index)
    observed_count = pd.Series(0.0, index=values.index)

    for lag in range(1, window + 1):
        lagged = grouped.shift(lag)
        present = lagged.notna()
        weight = weights[lag - 1]
        weighted_total = weighted_total.add(lagged.fillna(0.0) * weight)
        weight_total = weight_total.add(present.astype(float) * weight)
        observed_count = observed_count.add(present.astype(float))

    result = weighted_total / weight_total.replace(0.0, np.nan)
    return result.where(observed_count >= min_periods)


def build_rolling_features(
    long_frame: pd.DataFrame, cfg: FeatureConfig
) -> pd.DataFrame:
    """Compute decayed rolling xG, xGA, form points and rest days per team."""
    rolling_cfg = cfg.rolling
    frame = long_frame.sort_values(["team", "MatchDate"]).reset_index(drop=True)
    team = frame["team"]

    for source, target in (("xg", "xg_roll"), ("xga", "xga_roll"), ("points", "points_roll")):
        frame[target] = decayed_rolling_mean(
            frame[source],
            team,
            window=rolling_cfg.window,
            half_life=rolling_cfg.half_life,
            min_periods=rolling_cfg.min_periods,
        )

    previous_match = frame.groupby("team")["MatchDate"].shift(1)
    rest = (frame["MatchDate"] - previous_match).dt.days
    frame["rest_days"] = rest.clip(upper=cfg.max_rest_days)

    frame["matches_played"] = frame.groupby("team").cumcount()
    return frame


def pivot_team_features(frame: pd.DataFrame, long_frame: pd.DataFrame) -> pd.DataFrame:
    """Fold team-perspective rolling features back onto the fixture grid."""
    carried = [
        "match_id",
        "is_home",
        "xg_roll",
        "xga_roll",
        "points_roll",
        "rest_days",
        "matches_played",
    ]
    subset = long_frame[carried]

    home = subset.loc[subset["is_home"] == 1].drop(columns="is_home")
    away = subset.loc[subset["is_home"] == 0].drop(columns="is_home")

    home = home.rename(
        columns={
            "xg_roll": "xg_home_roll",
            "xga_roll": "xga_home_roll",
            "points_roll": "points_home_roll",
            "rest_days": "rest_days_home",
            "matches_played": "matches_played_home",
        }
    )
    away = away.rename(
        columns={
            "xg_roll": "xg_away_roll",
            "xga_roll": "xga_away_roll",
            "points_roll": "points_away_roll",
            "rest_days": "rest_days_away",
            "matches_played": "matches_played_away",
        }
    )

    merged = frame.merge(home, on="match_id", how="left", validate="one_to_one")
    merged = merged.merge(away, on="match_id", how="left", validate="one_to_one")
    return merged


def add_derived_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create interaction terms that encode the matchup rather than the teams."""
    frame = frame.copy()
    frame["xg_diff_home"] = frame["xg_home_roll"] - frame["xga_home_roll"]
    frame["xg_diff_away"] = frame["xg_away_roll"] - frame["xga_away_roll"]
    # Net attacking edge: what the home side creates against what the away
    # side concedes, less the mirror image.
    frame["xg_matchup_edge"] = (frame["xg_home_roll"] + frame["xga_away_roll"]) - (
        frame["xg_away_roll"] + frame["xga_home_roll"]
    )
    frame["form_diff"] = frame["points_home_roll"] - frame["points_away_roll"]
    frame["rest_diff"] = frame["rest_days_home"] - frame["rest_days_away"]
    frame["matchweek"] = frame.groupby("Season").cumcount() // 10 + 1
    return frame


def build_features(
    dataset: pd.DataFrame,
    cfg: FeatureConfig | None = None,
    burn_in_seasons: int = 1,
) -> tuple[pd.DataFrame, PoissonRegressor]:
    """Run the full feature pipeline.

    Parameters
    ----------
    dataset
        Output of :func:`src.data.build_dataset`.
    cfg
        Feature hyper-parameters.
    burn_in_seasons
        Number of leading seasons used to fit the shot-quality model and warm
        up Elo. These rows are flagged ``is_burn_in`` and dropped before
        training.

    Returns
    -------
    tuple
        The feature frame and the fitted shot-quality model (needed at
        inference time).
    """
    cfg = cfg or config.FEATURE_CONFIG
    frame = compute_elo_ratings(dataset, cfg)

    long_frame = to_team_perspective(frame)
    seasons = sorted(frame["Season"].unique())
    burn_in = set(seasons[:burn_in_seasons])

    shot_model = fit_shot_quality_model(long_frame.loc[long_frame["Season"].isin(burn_in)])
    long_frame = apply_shot_quality(long_frame, shot_model)
    long_frame = build_rolling_features(long_frame, cfg)

    frame = pivot_team_features(frame, long_frame)
    frame = add_derived_features(frame)
    frame["is_burn_in"] = frame["Season"].isin(burn_in)

    complete = frame[list(MODEL_FEATURES)].notna().all(axis=1)
    frame["is_modellable"] = complete & ~frame["is_burn_in"]

    logger.info(
        "Built %d features for %d fixtures (%d modellable).",
        len(MODEL_FEATURES),
        len(frame),
        int(frame["is_modellable"].sum()),
    )
    return frame, shot_model


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    from src.data import build_dataset

    features, _ = build_features(build_dataset())
    features.to_parquet(config.FEATURE_STORE_FILE, index=False)
    print(features.loc[features["is_modellable"], list(MODEL_FEATURES)].describe().T)


def build_team_snapshot(features: pd.DataFrame) -> dict[str, dict[str, Any]]:
    """Reduce the fixture-level feature store to one current state per team.

    Each fixture carries pre-match ratings for both sides, so a team's most
    recent appearance -- home or away -- holds the freshest view of its form.
    Reshaping to team perspective and taking the last row per team gives the
    state needed to score a hypothetical fixture.

    Values are pre-match as of that fixture, so they are one match stale.
    ``as_of`` travels with them so a caller can judge how current they are.
    """
    home = features[[
        "MatchDate", "HomeTeam", "elo_home", "xg_home_roll",
        "xga_home_roll", "points_home_roll",
    ]].rename(columns={
        "HomeTeam": "team", "elo_home": "elo", "xg_home_roll": "xg_roll",
        "xga_home_roll": "xga_roll", "points_home_roll": "points_roll",
    })
    away = features[[
        "MatchDate", "AwayTeam", "elo_away", "xg_away_roll",
        "xga_away_roll", "points_away_roll",
    ]].rename(columns={
        "AwayTeam": "team", "elo_away": "elo", "xg_away_roll": "xg_roll",
        "xga_away_roll": "xga_roll", "points_away_roll": "points_roll",
    })

    combined = (
        pd.concat([home, away], ignore_index=True)
        .dropna(subset=["team"])
        .sort_values("MatchDate")
    )
    latest = combined.groupby("team", as_index=True).last()

    return {
        str(team): {
            "elo": float(row["elo"]),
            "xg_roll": float(row["xg_roll"]),
            "xga_roll": float(row["xga_roll"]),
            "points_roll": float(row["points_roll"]),
            "as_of": row["MatchDate"].date().isoformat(),
        }
        for team, row in latest.iterrows()
    }
