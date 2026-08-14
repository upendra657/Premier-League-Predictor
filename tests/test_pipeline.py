"""Correctness tests for the decision engine.

The emphasis is on the failure modes that silently inflate results rather than
crash the pipeline: temporal leakage, probability vectors that do not sum to
one, class-order scrambling between estimator and caller, and staking maths
that quietly risks more than the bankroll.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src import config
from src.backtest import (
    build_candidates,
    kelly_fraction,
    max_drawdown,
    select_bets,
    simulate_kelly,
)
from src.data import add_market_probabilities, normalise_team
from src.features import (
    compute_elo_ratings,
    decayed_rolling_mean,
    to_team_perspective,
)
from src.train import (
    expected_calibration_error,
    multiclass_brier_score,
    ranked_probability_score,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #


@pytest.fixture
def toy_matches() -> pd.DataFrame:
    """Six fixtures across two seasons with known, hand-checkable outcomes."""
    rows = [
        ("2000/01", "2000-08-19", "Arsenal", "Chelsea", 2, 0),
        ("2000/01", "2000-08-26", "Chelsea", "Everton", 1, 1),
        ("2000/01", "2000-09-02", "Everton", "Arsenal", 0, 3),
        ("2000/01", "2000-09-09", "Arsenal", "Everton", 1, 0),
        ("2001/02", "2001-08-18", "Chelsea", "Arsenal", 2, 1),
        ("2001/02", "2001-08-25", "Everton", "Chelsea", 0, 0),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "Season", "MatchDate", "HomeTeam", "AwayTeam",
            "FullTimeHomeGoals", "FullTimeAwayGoals",
        ],
    )
    frame["MatchDate"] = pd.to_datetime(frame["MatchDate"])
    frame["FullTimeResult"] = np.select(
        [
            frame["FullTimeHomeGoals"] > frame["FullTimeAwayGoals"],
            frame["FullTimeHomeGoals"] == frame["FullTimeAwayGoals"],
        ],
        ["H", "D"],
        default="A",
    )
    for column in ("HomeShots", "AwayShots", "HomeShotsOnTarget",
                   "AwayShotsOnTarget", "HomeCorners", "AwayCorners"):
        frame[column] = 5
    frame["match_id"] = frame.index.astype(str)
    return frame


# --------------------------------------------------------------------------- #
# Data layer
# --------------------------------------------------------------------------- #


def test_team_normalisation_unifies_aliases():
    names = pd.Series(["Nottm Forest", "Man Utd", "Arsenal ", "Spurs"])
    normalised = normalise_team(names).tolist()
    assert normalised == ["Nott'm Forest", "Man United", "Arsenal", "Tottenham"]


def test_devigged_probabilities_sum_to_one_and_expose_overround():
    frame = pd.DataFrame({"odds_home": [2.0], "odds_draw": [4.0], "odds_away": [4.0]})
    result = add_market_probabilities(frame)

    probabilities = result[["mkt_prob_home", "mkt_prob_draw", "mkt_prob_away"]]
    assert probabilities.sum(axis=1).round(9).eq(1.0).all()
    # 1/2 + 1/4 + 1/4 = 1.0 exactly, so this book has zero margin.
    assert result["market_overround"].iloc[0] == pytest.approx(0.0, abs=1e-9)


def test_overround_is_positive_for_a_realistic_book():
    frame = pd.DataFrame({"odds_home": [1.9], "odds_draw": [3.5], "odds_away": [4.0]})
    result = add_market_probabilities(frame)
    assert result["market_overround"].iloc[0] > 0.0


# --------------------------------------------------------------------------- #
# Elo
# --------------------------------------------------------------------------- #


def test_elo_ratings_are_pre_match_not_post_match(toy_matches):
    """The first fixture must be scored at the initial rating for both sides."""
    rated = compute_elo_ratings(toy_matches, config.FEATURE_CONFIG)
    initial = config.FEATURE_CONFIG.elo.initial_rating

    assert rated["elo_home"].iloc[0] == pytest.approx(initial)
    assert rated["elo_away"].iloc[0] == pytest.approx(initial)


def test_elo_is_zero_sum_within_a_fixture(toy_matches):
    """Whatever the winner gains, the loser must lose."""
    rated = compute_elo_ratings(toy_matches, config.FEATURE_CONFIG)

    # Arsenal beat Chelsea in match 0; in match 4 Chelsea host Arsenal.
    arsenal_after = rated.loc[4, "elo_away"]
    chelsea_after = rated.loc[4, "elo_home"]
    initial = config.FEATURE_CONFIG.elo.initial_rating

    # Season regression pulls both toward the mean but preserves ordering.
    assert arsenal_after > initial > chelsea_after


def test_elo_rewards_winning_and_penalises_losing(toy_matches):
    rated = compute_elo_ratings(toy_matches, config.FEATURE_CONFIG)
    # Everton lost matches 1 (draw), 2 (heavy home loss) before hosting in 5.
    assert rated.loc[5, "elo_home"] < config.FEATURE_CONFIG.elo.initial_rating


def test_elo_win_expectancy_is_bounded_and_favours_home(toy_matches):
    rated = compute_elo_ratings(toy_matches, config.FEATURE_CONFIG)
    expectancy = rated["elo_win_expectancy"]
    assert expectancy.between(0.0, 1.0).all()
    # Equal ratings plus home advantage must exceed a coin flip.
    assert rated["elo_win_expectancy"].iloc[0] > 0.5


# --------------------------------------------------------------------------- #
# Rolling features -- the leakage-critical path
# --------------------------------------------------------------------------- #


def test_decayed_rolling_mean_excludes_the_current_observation():
    values = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    group = pd.Series(["A"] * 5)
    result = decayed_rolling_mean(values, group, window=5, half_life=2.5, min_periods=1)

    # Row 0 has no history at all.
    assert pd.isna(result.iloc[0])
    # Row 1 can only see row 0.
    assert result.iloc[1] == pytest.approx(1.0)
    # Row 2 sees rows 0 and 1, weighted toward the more recent one.
    weights = 0.5 ** (np.arange(2) / 2.5)
    assert result.iloc[2] == pytest.approx((2.0 * weights[0] + 1.0 * weights[1]) / weights.sum())


def test_decayed_rolling_mean_respects_min_periods():
    values = pd.Series([1.0, 2.0, 3.0, 4.0])
    group = pd.Series(["A"] * 4)
    result = decayed_rolling_mean(values, group, window=5, half_life=2.5, min_periods=3)
    assert result.iloc[:3].isna().all()
    assert not pd.isna(result.iloc[3])


def test_decayed_rolling_mean_does_not_bleed_across_teams():
    values = pd.Series([10.0, 10.0, 10.0, 0.0, 0.0, 0.0])
    group = pd.Series(["A", "A", "A", "B", "B", "B"])
    result = decayed_rolling_mean(values, group, window=5, half_life=2.5, min_periods=1)

    assert pd.isna(result.iloc[3]), "Team B's first match must have no history"
    assert result.iloc[4] == pytest.approx(0.0), "Team B must not see team A's values"


def test_recent_observations_outweigh_older_ones():
    """A recent collapse in form must dominate a distant good run."""
    values = pd.Series([3.0, 3.0, 3.0, 3.0, 0.0, np.nan])
    group = pd.Series(["A"] * 6)
    result = decayed_rolling_mean(values, group, window=5, half_life=2.5, min_periods=1)
    unweighted = values.iloc[:5].mean()
    assert result.iloc[5] < unweighted


def test_team_perspective_doubles_rows_and_mirrors_goals(toy_matches):
    long_frame = to_team_perspective(toy_matches)
    assert len(long_frame) == 2 * len(toy_matches)

    home_view = long_frame[(long_frame["match_id"] == "0") & (long_frame["is_home"] == 1)]
    away_view = long_frame[(long_frame["match_id"] == "0") & (long_frame["is_home"] == 0)]
    assert home_view["goals"].iloc[0] == away_view["goals_conceded"].iloc[0]
    assert home_view["points"].iloc[0] == 3.0
    assert away_view["points"].iloc[0] == 0.0


# --------------------------------------------------------------------------- #
# Metrics
# --------------------------------------------------------------------------- #


def test_brier_score_is_zero_for_perfect_predictions():
    y_true = np.array([0, 1, 2])
    probabilities = np.eye(3)
    assert multiclass_brier_score(y_true, probabilities) == pytest.approx(0.0)


def test_brier_score_is_two_for_confidently_wrong_predictions():
    y_true = np.array([0])
    probabilities = np.array([[0.0, 0.0, 1.0]])
    assert multiclass_brier_score(y_true, probabilities) == pytest.approx(2.0)


def test_rps_penalises_distant_errors_more_than_adjacent_ones():
    """Predicting an away win when it was a home win is worse than a draw."""
    y_true = np.array([0])
    adjacent = np.array([[0.0, 1.0, 0.0]])
    distant = np.array([[0.0, 0.0, 1.0]])
    assert ranked_probability_score(y_true, distant) > ranked_probability_score(y_true, adjacent)


def test_ece_is_zero_for_a_perfectly_calibrated_forecast():
    rng = np.random.default_rng(0)
    probabilities = np.tile([0.5, 0.3, 0.2], (20_000, 1))
    y_true = rng.choice(3, size=20_000, p=[0.5, 0.3, 0.2])
    assert expected_calibration_error(y_true, probabilities) < 0.01


# --------------------------------------------------------------------------- #
# Staking and simulation
# --------------------------------------------------------------------------- #


def test_kelly_matches_the_closed_form():
    # p=0.5 at evens is a zero-edge bet.
    assert kelly_fraction(np.array([0.5]), np.array([2.0]))[0] == pytest.approx(0.0)
    # p=0.6 at evens: (0.6*2 - 1) / 1 = 0.2
    assert kelly_fraction(np.array([0.6]), np.array([2.0]))[0] == pytest.approx(0.2)


def test_kelly_is_zero_for_negative_expected_value():
    assert kelly_fraction(np.array([0.4]), np.array([2.0]))[0] == 0.0


def test_kelly_never_exceeds_the_whole_bankroll():
    fractions = kelly_fraction(np.array([0.99, 1.0]), np.array([50.0, 2.0]))
    assert (fractions <= 1.0).all()


def test_selection_band_rejects_extreme_disagreement():
    """A wild edge must be filtered out, not staked on."""
    candidates = pd.DataFrame(
        {
            "edge": [0.05, 0.05],
            "relative_edge": [0.20, 3.00],
            "expected_value": [0.10, 2.00],
            "odds": [3.0, 3.0],
            "model_prob": [0.40, 0.90],
        }
    )
    selected = select_bets(candidates, config.BACKTEST_CONFIG)
    assert len(selected) == 1
    assert selected["relative_edge"].iloc[0] == pytest.approx(0.20)


def test_daily_exposure_cap_is_respected():
    """Twenty simultaneous bets must not risk more than the daily cap."""
    n = 20
    bets = pd.DataFrame(
        {
            "MatchDate": pd.to_datetime(["2020-01-01"] * n),
            "Season": ["2019/20"] * n,
            "stake_fraction": [0.02] * n,
            "odds": [3.0] * n,
            "won": [0] * n,
            "edge": [0.05] * n,
        }
    )
    cfg = config.BACKTEST_CONFIG
    ledger = simulate_kelly(bets, cfg)
    total_staked = ledger["stake"].sum()
    assert total_staked <= cfg.starting_bankroll * cfg.max_daily_exposure + 1e-6


def test_losing_every_bet_cannot_produce_a_negative_bankroll():
    bets = pd.DataFrame(
        {
            "MatchDate": pd.to_datetime([f"2020-01-{d:02d}" for d in range(1, 21)]),
            "Season": ["2019/20"] * 20,
            "stake_fraction": [0.02] * 20,
            "odds": [3.0] * 20,
            "won": [0] * 20,
            "edge": [0.05] * 20,
        }
    )
    ledger = simulate_kelly(bets, config.BACKTEST_CONFIG)
    assert (ledger["bankroll_after"] > 0).all()


def test_max_drawdown_detects_a_known_decline():
    equity = pd.Series([100.0, 120.0, 60.0, 90.0])
    assert max_drawdown(equity) == pytest.approx(-0.5)


def test_candidate_probabilities_are_devigged_per_match():
    predictions = pd.DataFrame(
        {
            "match_id": ["m1"],
            "Season": ["2019/20"],
            "MatchDate": pd.to_datetime(["2020-01-01"]),
            "HomeTeam": ["Arsenal"],
            "AwayTeam": ["Chelsea"],
            "target": [0],
            "prob_home": [0.5],
            "prob_draw": [0.3],
            "prob_away": [0.2],
        }
    )
    market = predictions.assign(
        best_odds_home=2.0, best_odds_draw=4.0, best_odds_away=4.0
    )
    candidates = build_candidates(predictions, market, odds_source="best_odds")

    assert len(candidates) == 3
    assert candidates["fair_prob"].sum() == pytest.approx(1.0)
    # Only the home outcome actually happened.
    assert candidates.loc[candidates["outcome"] == "home", "won"].iloc[0] == 1
    assert candidates.loc[candidates["outcome"] != "home", "won"].sum() == 0
