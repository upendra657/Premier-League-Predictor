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
from src.features import MODEL_FEATURES

logger = logging.getLogger(__name__)

#: Populated at startup; kept module-level so handlers stay dependency-free.
STATE: dict[str, Any] = {"model": None, "metadata": None, "features": None}


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

    elo_home: float = Field(1500.0, ge=1000.0, le=2200.0)
    elo_away: float = Field(1500.0, ge=1000.0, le=2200.0)

    xg_home_roll: float = Field(1.35, ge=0.0, le=6.0, description="Decayed rolling xG.")
    xg_away_roll: float = Field(1.35, ge=0.0, le=6.0)
    xga_home_roll: float = Field(1.35, ge=0.0, le=6.0, description="Decayed rolling xG against.")
    xga_away_roll: float = Field(1.35, ge=0.0, le=6.0)

    points_home_roll: float = Field(1.35, ge=0.0, le=3.0)
    points_away_roll: float = Field(1.35, ge=0.0, le=3.0)

    rest_days_home: int = Field(7, ge=0, le=14)
    rest_days_away: int = Field(7, ge=0, le=14)
    matchweek: int = Field(19, ge=1, le=38)

    def to_frame(self) -> pd.DataFrame:
        """Assemble the exact feature vector the trained model expects."""
        elo_diff = self.elo_home - self.elo_away
        home_advantage = config.FEATURE_CONFIG.elo.home_advantage
        scale = config.FEATURE_CONFIG.elo.scale

        row = {
            "elo_home": self.elo_home,
            "elo_away": self.elo_away,
            "elo_diff": elo_diff,
            "elo_win_expectancy": 1.0 / (1.0 + 10.0 ** (-(elo_diff + home_advantage) / scale)),
            "xg_home_roll": self.xg_home_roll,
            "xg_away_roll": self.xg_away_roll,
            "xga_home_roll": self.xga_home_roll,
            "xga_away_roll": self.xga_away_roll,
            "xg_diff_home": self.xg_home_roll - self.xga_home_roll,
            "xg_diff_away": self.xg_away_roll - self.xga_away_roll,
            "xg_matchup_edge": (self.xg_home_roll + self.xga_away_roll)
            - (self.xg_away_roll + self.xga_home_roll),
            "points_home_roll": self.points_home_roll,
            "points_away_roll": self.points_away_roll,
            "form_diff": self.points_home_roll - self.points_away_roll,
            "rest_days_home": self.rest_days_home,
            "rest_days_away": self.rest_days_away,
            "rest_diff": self.rest_days_home - self.rest_days_away,
            "matchweek": self.matchweek,
        }
        return pd.DataFrame([row])[list(MODEL_FEATURES)]


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
    return PredictionResponse(
        home_team=fixture.home_team,
        away_team=fixture.away_team,
        prob_home_win=round(float(probabilities[0]), 4),
        prob_draw=round(float(probabilities[1]), 4),
        prob_away_win=round(float(probabilities[2]), 4),
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

    prediction = PredictionResponse(
        home_team=request.fixture.home_team,
        away_team=request.fixture.away_team,
        prob_home_win=round(float(probabilities[0]), 4),
        prob_draw=round(float(probabilities[1]), 4),
        prob_away_win=round(float(probabilities[2]), 4),
    )
    return ValueResponse(
        prediction=prediction,
        market_overround=round(float(booksum - 1.0), 4),
        recommendations=recommendations,
    )


if __name__ == "__main__":  # pragma: no cover
    import uvicorn

    uvicorn.run("src.main:app", host="0.0.0.0", port=8000, reload=False)
