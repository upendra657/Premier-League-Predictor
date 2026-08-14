"""FastAPI inference service for the Premier League Decision Engine.

Exposes calibrated match probabilities and, when a price is supplied, the
staking recommendation that follows from them. The service is deliberately
thin: it loads artifacts once at startup and does no feature computation
beyond assembling the vector the model expects.

Endpoints
---------
``GET  /health``   Liveness and readiness, including artifact status.
``POST /predict``  Calibrated Home/Draw/Away probabilities for one fixture.
``POST /value``    Probabilities plus edge and Kelly stake against given odds.
``GET  /model``    Metadata card for the loaded model.
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, computed_field

from src import config
from src.backtest import kelly_fraction
from src.features import MODEL_FEATURES, build_team_snapshot

logger = logging.getLogger(__name__)

#: Populated at startup; kept module-level so handlers stay dependency-free.
STATE: dict[str, Any] = {
    "model": None,
    "metadata": None,
    "features": None,
    "teams": {},
}

#: Fallback when a team has no history and the caller supplied nothing.
LEAGUE_AVERAGE = {
    "elo": 1500.0,
    "xg_roll": 1.35,
    "xga_roll": 1.35,
    "points_roll": 1.35,
}


def build_feature_frame(
    values: dict[str, float], rest_home: int, rest_away: int, matchweek: int
) -> pd.DataFrame:
    """Derive the full feature vector from resolved per-team ratings."""
    elo_diff = values["elo_home"] - values["elo_away"]
    home_advantage = config.FEATURE_CONFIG.elo.home_advantage
    scale = config.FEATURE_CONFIG.elo.scale

    row = {
        "elo_home": values["elo_home"],
        "elo_away": values["elo_away"],
        "elo_diff": elo_diff,
        "elo_win_expectancy": 1.0 / (1.0 + 10.0 ** (-(elo_diff + home_advantage) / scale)),
        "xg_home_roll": values["xg_home_roll"],
        "xg_away_roll": values["xg_away_roll"],
        "xga_home_roll": values["xga_home_roll"],
        "xga_away_roll": values["xga_away_roll"],
        "xg_diff_home": values["xg_home_roll"] - values["xga_home_roll"],
        "xg_diff_away": values["xg_away_roll"] - values["xga_away_roll"],
        "xg_matchup_edge": (values["xg_home_roll"] + values["xga_away_roll"])
        - (values["xg_away_roll"] + values["xga_home_roll"]),
        "points_home_roll": values["points_home_roll"],
        "points_away_roll": values["points_away_roll"],
        "form_diff": values["points_home_roll"] - values["points_away_roll"],
        "rest_days_home": rest_home,
        "rest_days_away": rest_away,
        "rest_diff": rest_home - rest_away,
        "matchweek": matchweek,
    }
    return pd.DataFrame([row])[list(MODEL_FEATURES)]


def resolve_team(name: str) -> dict[str, Any]:
    """Look up a team's current state, or 404 with the names that do exist."""
    teams = STATE["teams"]
    if not teams:
        return {**LEAGUE_AVERAGE, "as_of": None}
    if name not in teams:
        raise HTTPException(
            status_code=404,
            detail={
                "message": f"Unknown team {name!r}.",
                "hint": "Team names follow football-data.co.uk conventions.",
                "known_teams": sorted(teams),
            },
        )
    return teams[name]


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once, at process start rather than per request."""
    import joblib

    if config.MODEL_FILE.exists():
        bundle = joblib.load(config.MODEL_FILE)
        STATE["model"] = bundle["model"]
        STATE["features"] = bundle["features"]
        logger.info("Loaded model with %d features.", len(bundle["features"]))
    else:
        logger.warning("No model artifact at %s; /predict will 503.", config.MODEL_FILE)

    if config.METADATA_FILE.exists():
        STATE["metadata"] = json.loads(config.METADATA_FILE.read_text())

    # Prefer the shipped snapshot: the inference image carries artifacts/ but
    # not data/, so reading the feature store here would work in development
    # and silently degrade to league averages inside the container.
    if config.TEAM_SNAPSHOT_FILE.exists():
        STATE["teams"] = json.loads(config.TEAM_SNAPSHOT_FILE.read_text())
        logger.info("Loaded current form for %d teams (artifact).", len(STATE["teams"]))
    elif config.FEATURE_STORE_FILE.exists():
        STATE["teams"] = build_team_snapshot(pd.read_parquet(config.FEATURE_STORE_FILE))
        logger.info("Loaded current form for %d teams (feature store).", len(STATE["teams"]))
    else:
        logger.warning(
            "No team snapshot at %s and no feature store at %s; predictions "
            "fall back to league averages unless the caller supplies ratings.",
            config.TEAM_SNAPSHOT_FILE,
            config.FEATURE_STORE_FILE,
        )

    yield
    STATE.clear()


app = FastAPI(
    title="Premier League Decision Engine",
    description=(
        "Calibrated match outcome probabilities and expected-value staking "
        "recommendations for English Premier League fixtures."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class FixtureFeatures(BaseModel):
    """Pre-match state for a single fixture.

    Every field is information that exists before kick-off. Defaults reflect
    a league-average, evenly matched fixture so callers can supply only the
    signals they actually have.
    """

    home_team: str = Field(..., examples=["Arsenal"])
    away_team: str = Field(..., examples=["Chelsea"])

    # Every rating below is optional. Left unset, it is looked up from the
    # team's most recent fixture in the feature store; set, it overrides that
    # lookup, which is what makes "what if Arsenal were rated 1700" answerable.
    elo_home: float | None = Field(None, ge=1000.0, le=2200.0)
    elo_away: float | None = Field(None, ge=1000.0, le=2200.0)

    xg_home_roll: float | None = Field(None, ge=0.0, le=6.0, description="Decayed rolling xG.")
    xg_away_roll: float | None = Field(None, ge=0.0, le=6.0)
    xga_home_roll: float | None = Field(
        None, ge=0.0, le=6.0, description="Decayed rolling xG against."
    )
    xga_away_roll: float | None = Field(None, ge=0.0, le=6.0)

    points_home_roll: float | None = Field(None, ge=0.0, le=3.0)
    points_away_roll: float | None = Field(None, ge=0.0, le=3.0)

    rest_days_home: int = Field(7, ge=0, le=14)
    rest_days_away: int = Field(7, ge=0, le=14)
    matchweek: int = Field(19, ge=1, le=38)

    def resolve(self) -> tuple[dict[str, float], dict[str, Any]]:
        """Merge caller-supplied ratings over each team's stored current form.

        Returns the resolved values and a provenance record naming, per side,
        whether the numbers came from the request or from the feature store.
        """
        home_state = resolve_team(self.home_team)
        away_state = resolve_team(self.away_team)

        supplied = self.model_dump(exclude_none=True)
        resolved = {
            "elo_home": self.elo_home if self.elo_home is not None else home_state["elo"],
            "elo_away": self.elo_away if self.elo_away is not None else away_state["elo"],
            "xg_home_roll": self.xg_home_roll
            if self.xg_home_roll is not None else home_state["xg_roll"],
            "xg_away_roll": self.xg_away_roll
            if self.xg_away_roll is not None else away_state["xg_roll"],
            "xga_home_roll": self.xga_home_roll
            if self.xga_home_roll is not None else home_state["xga_roll"],
            "xga_away_roll": self.xga_away_roll
            if self.xga_away_roll is not None else away_state["xga_roll"],
            "points_home_roll": self.points_home_roll
            if self.points_home_roll is not None else home_state["points_roll"],
            "points_away_roll": self.points_away_roll
            if self.points_away_roll is not None else away_state["points_roll"],
        }
        overridden = sorted(k for k in resolved if k in supplied)
        provenance = {
            "home_form_as_of": home_state.get("as_of"),
            "away_form_as_of": away_state.get("as_of"),
            "overridden_fields": overridden,
        }
        return resolved, provenance

    def to_frame(self) -> pd.DataFrame:
        """Assemble the exact feature vector the trained model expects."""
        values, _ = self.resolve()
        return build_feature_frame(values, self.rest_days_home,
                                   self.rest_days_away, self.matchweek)


class MarketOdds(BaseModel):
    """Decimal odds quoted for each outcome."""

    home: float = Field(..., gt=1.0, le=100.0)
    draw: float = Field(..., gt=1.0, le=100.0)
    away: float = Field(..., gt=1.0, le=100.0)


class ValueRequest(BaseModel):
    """A fixture together with the prices available on it."""

    fixture: FixtureFeatures
    odds: MarketOdds


class PredictionResponse(BaseModel):
    """Calibrated outcome probabilities."""

    home_team: str
    away_team: str
    prob_home_win: float
    prob_draw: float
    prob_away_win: float
    form_as_of: dict[str, Any] | None = Field(
        None,
        description=(
            "Which fixture each side's ratings were taken from, and any fields "
            "the request overrode."
        ),
    )

    @computed_field  # type: ignore[misc]
    @property
    def most_likely(self) -> str:
        """Modal outcome -- reported for convenience, not for staking."""
        options = {
            "home_win": self.prob_home_win,
            "draw": self.prob_draw,
            "away_win": self.prob_away_win,
        }
        return max(options, key=options.get)


class Recommendation(BaseModel):
    """Staking decision for one selection."""

    outcome: str
    odds: float
    model_probability: float
    fair_market_probability: float
    edge: float
    relative_edge: float
    expected_value: float
    kelly_stake_fraction: float
    recommended: bool


class ValueResponse(BaseModel):
    """Probabilities plus the value assessment for all three selections."""

    prediction: PredictionResponse
    market_overround: float
    recommendations: list[Recommendation]


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #


def _require_model():
    """Return the loaded estimator or fail with a clear 503."""
    if STATE["model"] is None:
        raise HTTPException(
            status_code=503,
            detail="Model artifact unavailable. Run `python -m src.train` first.",
        )
    return STATE["model"]


def _predict_probabilities(fixture: FixtureFeatures) -> np.ndarray:
    """Score one fixture and return probabilities in [H, D, A] order."""
    model = _require_model()
    raw = model.predict_proba(fixture.to_frame())
    order = [list(model.classes_).index(i) for i in range(3)]
    return raw[:, order].ravel()


def round_to_unit(probabilities: np.ndarray, places: int = 4) -> list[float]:
    """Round a probability vector so the values sum to one at ``places``.

    Rounding each element independently lets the total drift off 1.0 -- three
    values ending in ...5 all round down and the vector sums to 0.9999. A
    client computing one outcome as ``1 - other - other`` then disagrees with
    the value served. The largest-remainder method assigns the leftover unit to
    whichever element was rounded down hardest, so no unit goes missing.

    The guarantee is exact in decimal, not in binary: the integer counts of
    ``10**-places`` units sum to exactly ``10**places``. Summing the returned
    floats can still land a few ulps off 1.0, because values like 0.3333 have
    no exact binary representation -- assert ``round(sum(...), places) == 1.0``
    rather than ``sum(...) == 1.0``.
    """
    scale = 10**places
    # float64 throughout: XGBoost hands back float32, and doing the arithmetic
    # at that precision leaves the total off by ~1e-8 even after correction.
    scaled = np.asarray(probabilities, dtype=np.float64) * scale
    floors = np.floor(scaled)
    shortfall = int(round(scale - floors.sum()))

    if shortfall > 0:
        # Hand the spare units to the largest fractional parts, biggest first.
        for index in np.argsort(-(scaled - floors))[:shortfall]:
            floors[index] += 1

    return [int(value) / scale for value in floors]


@app.get("/health", tags=["ops"])
def health() -> dict[str, Any]:
    """Liveness probe reporting whether the service can actually serve."""
    ready = STATE["model"] is not None
    return {
        "status": "ok" if ready else "degraded",
        "model_loaded": ready,
        "n_features": len(STATE["features"]) if STATE["features"] else 0,
        "version": app.version,
    }


@app.get("/model", tags=["ops"])
def model_card() -> dict[str, Any]:
    """Expose the model metadata card, including walk-forward metrics."""
    if STATE["metadata"] is None:
        raise HTTPException(status_code=503, detail="Model metadata unavailable.")
    return STATE["metadata"]


@app.post("/predict", response_model=PredictionResponse, tags=["inference"])
def predict(fixture: FixtureFeatures) -> PredictionResponse:
    """Return calibrated Home/Draw/Away probabilities for a fixture."""
    probabilities = _predict_probabilities(fixture)
    _, provenance = fixture.resolve()
    home, draw, away = round_to_unit(probabilities)
    return PredictionResponse(
        home_team=fixture.home_team,
        away_team=fixture.away_team,
        prob_home_win=home,
        prob_draw=draw,
        prob_away_win=away,
        form_as_of=provenance,
    )


@app.post("/value", response_model=ValueResponse, tags=["inference"])
def value(request: ValueRequest) -> ValueResponse:
    """Assess every selection on a fixture against the quoted prices."""
    probabilities = _predict_probabilities(request.fixture)
    quoted = np.array([request.odds.home, request.odds.draw, request.odds.away])

    implied = 1.0 / quoted
    booksum = implied.sum()
    fair = implied / booksum

    strategy = config.BACKTEST_CONFIG.strategy
    edges = probabilities - fair
    relative_edges = probabilities / fair - 1.0
    expected_values = probabilities * quoted - 1.0
    kelly = kelly_fraction(probabilities, quoted) * config.BACKTEST_CONFIG.kelly_fraction
    kelly = np.minimum(kelly, config.BACKTEST_CONFIG.max_stake_fraction)

    recommendations = []
    for i, outcome in enumerate(("home", "draw", "away")):
        qualifies = bool(
            edges[i] >= strategy.min_edge
            and strategy.min_relative_edge <= relative_edges[i] <= strategy.max_relative_edge
            and strategy.min_odds <= quoted[i] <= strategy.max_odds
            and expected_values[i] > 0.0
        )
        recommendations.append(
            Recommendation(
                outcome=outcome,
                odds=float(quoted[i]),
                model_probability=round(float(probabilities[i]), 4),
                fair_market_probability=round(float(fair[i]), 4),
                edge=round(float(edges[i]), 4),
                relative_edge=round(float(relative_edges[i]), 4),
                expected_value=round(float(expected_values[i]), 4),
                kelly_stake_fraction=round(float(kelly[i]) if qualifies else 0.0, 4),
                recommended=qualifies,
            )
        )

    _, provenance = request.fixture.resolve()
    home, draw, away = round_to_unit(probabilities)
    prediction = PredictionResponse(
        home_team=request.fixture.home_team,
        away_team=request.fixture.away_team,
        prob_home_win=home,
        prob_draw=draw,
        prob_away_win=away,
        form_as_of=provenance,
    )
    return ValueResponse(
        prediction=prediction,
        market_overround=round(float(booksum - 1.0), 4),
        recommendations=recommendations,
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)
