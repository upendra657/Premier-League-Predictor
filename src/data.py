"""Data ingestion and integration layer.

Builds the canonical match spine from the curated Premier League results file
and enriches it with 1X2 closing odds from a public football-data.co.uk
mirror. The output of :func:`build_dataset` is the single source of truth that
every downstream module consumes.

Design notes
------------
* The spine is authoritative for results and match statistics; the market file
  is joined in for odds only. This keeps label provenance clean.
* The join is validated, not assumed: :func:`build_dataset` raises if odds
  coverage falls below a configurable floor, so a silently broken upstream
  source fails loudly rather than producing a meaningless backtest.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src import config

logger = logging.getLogger(__name__)

#: Columns the spine file must provide.
REQUIRED_SPINE_COLUMNS: tuple[str, ...] = (
    "Season",
    "MatchDate",
    "HomeTeam",
    "AwayTeam",
    "FullTimeHomeGoals",
    "FullTimeAwayGoals",
    "FullTimeResult",
    "HomeShots",
    "AwayShots",
    "HomeShotsOnTarget",
    "AwayShotsOnTarget",
    "HomeCorners",
    "AwayCorners",
)

#: Canonical spellings for clubs whose names differ between data providers.
TEAM_NAME_ALIASES: dict[str, str] = {
    "Nottm Forest": "Nott'm Forest",
    "Nottingham Forest": "Nott'm Forest",
    "Man Utd": "Man United",
    "Manchester United": "Man United",
    "Manchester City": "Man City",
    "Spurs": "Tottenham",
    "Sheffield Utd": "Sheffield United",
    "Sheff Utd": "Sheffield United",
    "Wolverhampton": "Wolves",
    "Middlesboro": "Middlesbrough",
    "West Bromwich Albion": "West Brom",
    "Newcastle United": "Newcastle",
}


def normalise_team(series: pd.Series) -> pd.Series:
    """Map provider-specific club spellings onto a canonical vocabulary."""
    cleaned = series.astype("string").str.strip()
    return cleaned.replace(TEAM_NAME_ALIASES)


def load_match_spine(path: Path | None = None) -> pd.DataFrame:
    """Load the curated results file that anchors the whole pipeline.

    Parameters
    ----------
    path
        Location of the spine CSV. Defaults to ``config.MATCH_SPINE_FILE``.

    Returns
    -------
    pandas.DataFrame
        One row per fixture, sorted chronologically, with a stable
        ``match_id`` and canonical team names.
    """
    path = path or config.MATCH_SPINE_FILE
    if not path.exists():
        raise FileNotFoundError(
            f"Match spine not found at {path}. Place the curated results "
            "CSV there before running the pipeline."
        )

    frame = pd.read_csv(path)
    missing = set(REQUIRED_SPINE_COLUMNS) - set(frame.columns)
    if missing:
        raise ValueError(f"Spine file is missing required columns: {sorted(missing)}")

    frame["MatchDate"] = pd.to_datetime(frame["MatchDate"], errors="coerce")
    if frame["MatchDate"].isna().any():
        bad = int(frame["MatchDate"].isna().sum())
        raise ValueError(f"{bad} rows in the spine have unparseable MatchDate values.")

    frame["HomeTeam"] = normalise_team(frame["HomeTeam"])
    frame["AwayTeam"] = normalise_team(frame["AwayTeam"])

    invalid_results = set(frame["FullTimeResult"].unique()) - set(config.CLASS_LABELS)
    if invalid_results:
        raise ValueError(f"Unexpected FullTimeResult values: {sorted(invalid_results)}")

    frame = frame.sort_values(["MatchDate", "HomeTeam"]).reset_index(drop=True)
    frame["match_id"] = (
        frame["MatchDate"].dt.strftime("%Y%m%d")
        + "_"
        + frame["HomeTeam"].str.replace(r"\W", "", regex=True)
        + "_"
        + frame["AwayTeam"].str.replace(r"\W", "", regex=True)
    )

    logger.info(
        "Loaded %d fixtures spanning %s to %s (%d seasons).",
        len(frame),
        frame["Season"].iloc[0],
        frame["Season"].iloc[-1],
        frame["Season"].nunique(),
    )
    return frame


def fetch_market_odds(
    path: Path | None = None,
    url: str | None = None,
    force_download: bool = False,
) -> pd.DataFrame:
    """Return 1X2 market odds for the Premier League, downloading if needed.

    The remote file covers every major European division; only the configured
    division is retained. Results are cached to disk so repeat runs are offline.
    """
    path = path or config.MARKET_ODDS_FILE
    url = url or config.MARKET_ODDS_URL

    if path.exists() and not force_download:
        logger.info("Using cached market odds at %s.", path)
        return pd.read_csv(path, parse_dates=["MatchDate"])

    logger.info("Downloading market odds from %s ...", url)
    raw = pd.read_csv(url, low_memory=False)
    odds = _extract_division_odds(raw)
    odds.to_csv(path, index=False)
    logger.info("Cached %d odds rows to %s.", len(odds), path)
    return odds


def _extract_division_odds(raw: pd.DataFrame) -> pd.DataFrame:
    """Slice the multi-division feed down to tidy 1X2 odds for one league."""
    odds = raw.loc[raw["Division"] == config.DIVISION_CODE].copy()
    odds["MatchDate"] = pd.to_datetime(odds["MatchDate"], errors="coerce")
    odds["HomeTeam"] = normalise_team(odds["HomeTeam"])
    odds["AwayTeam"] = normalise_team(odds["AwayTeam"])

    column_map = {
        "OddHome": "odds_home",
        "OddDraw": "odds_draw",
        "OddAway": "odds_away",
        "MaxHome": "best_odds_home",
        "MaxDraw": "best_odds_draw",
        "MaxAway": "best_odds_away",
    }
    available = {src: dst for src, dst in column_map.items() if src in odds.columns}
    keep = ["MatchDate", "HomeTeam", "AwayTeam", *available]
    return odds[keep].rename(columns=available).dropna(subset=["MatchDate"])


def attach_market_odds(
    spine: pd.DataFrame,
    odds: pd.DataFrame,
    min_coverage: float = 0.95,
) -> pd.DataFrame:
    """Left-join odds onto the spine and assert the join actually worked.

    Raises
    ------
    ValueError
        If fewer than ``min_coverage`` of fixtures receive home odds, which
        almost always indicates a team-name mismatch rather than genuinely
        missing prices.
    """
    key = ["MatchDate", "HomeTeam", "AwayTeam"]
    odds = odds.drop_duplicates(subset=key, keep="first")
    merged = spine.merge(odds, on=key, how="left", validate="one_to_one")

    coverage = merged["odds_home"].notna().mean()
    if coverage < min_coverage:
        unmatched = merged.loc[merged["odds_home"].isna(), "HomeTeam"].value_counts()
        raise ValueError(
            f"Odds coverage {coverage:.1%} is below the {min_coverage:.0%} floor. "
            f"Most frequent unmatched home teams: {unmatched.head(5).to_dict()}"
        )

    logger.info("Attached market odds to %.1f%% of fixtures.", coverage * 100)
    return merged


def add_market_probabilities(frame: pd.DataFrame) -> pd.DataFrame:
    """Convert decimal odds into overround-free (de-vigged) probabilities.

    Bookmaker odds imply probabilities that sum to more than one; the excess
    is the overround (vig). Normalising by the booksum removes it under the
    standard proportional assumption, yielding the market's fair estimate --
    the benchmark any model must beat to be economically useful.
    """
    frame = frame.copy()
    odds_columns = ["odds_home", "odds_draw", "odds_away"]
    prob_columns = ["mkt_prob_home", "mkt_prob_draw", "mkt_prob_away"]

    with np.errstate(divide="ignore", invalid="ignore"):
        raw_probs = 1.0 / frame[odds_columns].to_numpy(dtype=float)

    booksum = raw_probs.sum(axis=1, keepdims=True)
    frame["market_overround"] = booksum.ravel() - 1.0
    frame[prob_columns] = raw_probs / booksum
    return frame


def build_dataset(force_download: bool = False) -> pd.DataFrame:
    """Produce the fully integrated match dataset used by the pipeline."""
    spine = load_match_spine()
    odds = fetch_market_odds(force_download=force_download)
    merged = attach_market_odds(spine, odds)
    enriched = add_market_probabilities(merged)
    enriched["target"] = enriched["FullTimeResult"].map(config.CLASS_TO_INDEX)
    return enriched


if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    dataset = build_dataset()
    print(dataset[["MatchDate", "HomeTeam", "AwayTeam", "FullTimeResult",
                   "odds_home", "mkt_prob_home", "market_overround"]].tail(5))
    print(f"\nRows: {len(dataset):,} | Odds coverage: "
          f"{dataset['odds_home'].notna().mean():.1%}")
