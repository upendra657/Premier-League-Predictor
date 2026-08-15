"""Tests for the inference service.

The emphasis is on the failure modes that return 200 and look plausible: a
prediction that silently ignores which teams are playing, and a probability
vector that does not sum to one.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from src.main import app, build_team_snapshot, round_to_unit

FIXTURE = {
    "home_team": "Arsenal",
    "away_team": "Chelsea",
    "match_date": "2025-08-16",
    "season": "2025/26",
}


@pytest.fixture(scope="module")
def client():
    """A client with lifespan run, so artifacts are actually loaded."""
    with TestClient(app) as test_client:
        yield test_client


def _served(client, **overrides):
    payload = {**FIXTURE, **overrides}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200, response.text
    return response.json()


# --------------------------------------------------------------------------- #
# Probability rounding
# --------------------------------------------------------------------------- #


def test_rounding_preserves_total_where_naive_rounding_loses_a_unit():
    """Three equal thirds each round down; the lost unit must be given back."""
    rounded = round_to_unit(np.array([1 / 3, 1 / 3, 1 / 3]))
    assert round(sum(rounded), 4) == 1.0
    assert sorted(rounded) == [0.3333, 0.3333, 0.3334]


def test_rounding_holds_for_float32_input():
    """XGBoost returns float32; the correction must survive that precision."""
    rng = np.random.default_rng(0)
    for _ in range(500):
        vector = rng.dirichlet([1, 1, 1]).astype(np.float32)
        assert round(sum(round_to_unit(vector)), 4) == 1.0


def test_rounding_leaves_an_exact_vector_untouched():
    assert round_to_unit(np.array([0.5, 0.25, 0.25])) == [0.5, 0.25, 0.25]


# --------------------------------------------------------------------------- #
# Team snapshot
# --------------------------------------------------------------------------- #


def test_snapshot_takes_each_team_latest_appearance():
    """A team's state must come from its most recent fixture, home or away."""
    frame = pd.DataFrame({
        "MatchDate": pd.to_datetime(["2024-01-01", "2024-02-01", "2024-03-01"]),
        "HomeTeam": ["Arsenal", "Chelsea", "Arsenal"],
        "AwayTeam": ["Chelsea", "Arsenal", "Everton"],
        "elo_home": [1500.0, 1600.0, 1700.0],
        "elo_away": [1400.0, 1450.0, 1300.0],
        "xg_home_roll": [1.5, 1.6, 1.7],
        "xg_away_roll": [1.1, 1.2, 1.3],
        "xga_home_roll": [1.0, 1.1, 1.2],
        "xga_away_roll": [1.4, 1.5, 1.6],
        "points_home_roll": [2.0, 2.1, 2.2],
        "points_away_roll": [1.0, 1.1, 1.2],
    })
    snapshot = build_team_snapshot(frame)

    # Arsenal last played at home on 2024-03-01 with elo_home 1700.
    assert snapshot["Arsenal"]["elo"] == pytest.approx(1700.0)
    assert snapshot["Arsenal"]["as_of"] == "2024-03-01"
    # Chelsea last played at home on 2024-02-01 with elo_home 1600.
    assert snapshot["Chelsea"]["elo"] == pytest.approx(1600.0)
    # Everton appears only as an away side.
    assert snapshot["Everton"]["elo"] == pytest.approx(1300.0)


# --------------------------------------------------------------------------- #
# Endpoints
# --------------------------------------------------------------------------- #


def test_health_reports_a_loaded_model(client):
    body = client.get("/health").json()
    assert body["model_loaded"] is True
    assert body["n_features"] == 18


def test_served_probabilities_sum_to_one(client):
    body = _served(client)
    total = body["prob_home_win"] + body["prob_draw"] + body["prob_away_win"]
    assert round(total, 4) == 1.0


def test_team_identity_changes_the_prediction(client):
    """The regression that matters: names must not be decorative."""
    strong = _served(client, home_team="Liverpool", away_team="Burnley")
    weak = _served(client, home_team="Burnley", away_team="Liverpool")
    assert strong["prob_home_win"] != weak["prob_home_win"]
    assert strong["prob_home_win"] > weak["prob_home_win"]


def test_home_advantage_is_visible_when_teams_swap(client):
    """The same pairing reversed must not produce the same home probability."""
    forward = _served(client)
    reversed_ = _served(client, home_team="Chelsea", away_team="Arsenal")
    assert forward["prob_home_win"] != reversed_["prob_home_win"]


def test_unknown_team_is_rejected_with_the_valid_names(client):
    response = client.post("/predict", json={**FIXTURE, "home_team": "Nonexistent FC"})
    assert response.status_code == 404
    detail = response.json()["detail"]
    assert "Nonexistent FC" in detail["message"]
    assert "Arsenal" in detail["known_teams"]


def test_explicit_rating_overrides_the_stored_form(client):
    baseline = _served(client)
    boosted = _served(client, elo_home=2100)
    assert boosted["prob_home_win"] > baseline["prob_home_win"]
    assert boosted["form_as_of"]["overridden_fields"] == ["elo_home"]


def test_provenance_names_the_source_fixture(client):
    body = _served(client)
    assert body["form_as_of"]["home_form_as_of"] is not None
    assert body["form_as_of"]["overridden_fields"] == []


def test_value_endpoint_returns_a_recommendation_per_outcome(client):
    response = client.post(
        "/value",
        json={"fixture": FIXTURE, "odds": {"home": 2.10, "draw": 3.60, "away": 3.40}},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["recommendations"]) == 3
    assert body["market_overround"] > 0.0

    nested = body["prediction"]
    total = nested["prob_home_win"] + nested["prob_draw"] + nested["prob_away_win"]
    assert round(total, 4) == 1.0
    # /value must report provenance too, not just /predict.
    assert nested["form_as_of"]["home_form_as_of"] is not None


def test_relegated_team_form_is_stale_but_labelled(client):
    """Burnley left the league in 2024; the response must not hide that."""
    body = _served(client, home_team="Liverpool", away_team="Burnley")
    assert body["form_as_of"]["away_form_as_of"] < body["form_as_of"]["home_form_as_of"]
