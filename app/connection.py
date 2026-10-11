"""Dual-Mode Resilient Data Access Layer (Zero Cold-Start Crashes).

Provides seamless failover between Live Database mode and Certified Snapshot
Cache mode (Apache Parquet) for Streamlit Community Cloud deployment.
"""

from __future__ import annotations

import os
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load local environment variables if available
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
CACHE_DIR = BASE_DIR / "cache"


@dataclass
class ConnectionInfo:
    mode: str  # 'live' or 'snapshot'
    label: str
    db_type: str
    role: str
    latency_ms: float
    error_message: Optional[str] = None
    is_live: bool = False


def _check_live_db() -> tuple[bool, str, str, float, Optional[str]]:
    """Test connection to live database if configured via st.secrets or .env."""
    start_time = time.perf_counter()

    # 1. Check for Streamlit secrets configuration
    secrets_db = None
    role = "ANALYST_READONLY"
    db_type = "SQLite / Analytics DB"

    try:
        if hasattr(st, "secrets") and len(st.secrets) > 0:
            if "database" in st.secrets:
                secrets_db = st.secrets["database"]
                db_type = secrets_db.get("type", "PostgreSQL / Cloud DB")
                role = secrets_db.get("role", "ANALYST_READONLY")
    except Exception:
        pass

    # 2. Check for environment variables
    env_db_path = os.getenv("EPL_DB_PATH") or os.getenv("DATABASE_URL")

    # If explicitly configured live DB exists, test it
    candidate_db = None
    if secrets_db and "path" in secrets_db:
        candidate_db = Path(secrets_db["path"])
    elif env_db_path:
        candidate_db = Path(env_db_path)
    else:
        # Check if local analytics SQLite database is active
        local_db = CACHE_DIR / "epl_analytics.db"
        if local_db.exists() and os.getenv("ENABLE_LIVE_MODE", "true").lower() == "true":
            candidate_db = local_db
            db_type = "Local SQLite Analytics"
            role = "ANALYST_REPORTER"

    if candidate_db and candidate_db.exists():
        try:
            conn = sqlite3.connect(f"file:{candidate_db}?mode=ro", uri=True, timeout=2.0)
            cursor = conn.cursor()
            cursor.execute("SELECT count(*) FROM fct_bet_ledger")
            _ = cursor.fetchone()
            conn.close()
            latency = (time.perf_counter() - start_time) * 1000
            return True, f"Live {db_type}", role, latency, None
        except Exception as exc:
            latency = (time.perf_counter() - start_time) * 1000
            return False, "Certified Snapshot Cache", "READONLY", latency, str(exc)

    # No live DB configured -> Snapshot failover
    latency = (time.perf_counter() - start_time) * 1000
    return False, "Certified Snapshot Cache", "READONLY", latency, None


@st.cache_data(ttl=600, show_spinner=False)
def get_connection_info() -> ConnectionInfo:
    """Evaluate connection state and cache status for 10 minutes."""
    is_live, label, role, latency, err = _check_live_db()
    mode = "live" if is_live else "snapshot"
    db_type = label if is_live else "Apache Parquet (PyArrow Native)"
    return ConnectionInfo(
        mode=mode,
        label=label,
        db_type=db_type,
        role=role,
        latency_ms=round(latency, 2),
        error_message=err,
        is_live=is_live,
    )


# --------------------------------------------------------------------------- #
# Cached Data Loaders (Sub-Second Performance with PyArrow)
# --------------------------------------------------------------------------- #


@st.cache_data(ttl=600, show_spinner=False)
def load_bet_ledger() -> pd.DataFrame:
    """Load the complete 2,154 walk-forward betting simulation ledger."""
    info = get_connection_info()
    if info.is_live:
        db_path = CACHE_DIR / "epl_analytics.db"
        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                df = pd.read_sql("SELECT * FROM fct_bet_ledger", conn)
                df["match_date"] = pd.to_datetime(df["match_date"])
                return df
        except Exception:
            pass  # Failover to parquet snapshot

    parquet_path = CACHE_DIR / "fct_bet_ledger.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        df["match_date"] = pd.to_datetime(df["match_date"])
        return df

    return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_match_candidates() -> pd.DataFrame:
    """Load all 13,527 scored candidate matches across 25 seasons."""
    info = get_connection_info()
    if info.is_live:
        db_path = CACHE_DIR / "epl_analytics.db"
        try:
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as conn:
                df = pd.read_sql("SELECT * FROM fct_match_candidates", conn)
                df["match_date"] = pd.to_datetime(df["match_date"])
                return df
        except Exception:
            pass

    parquet_path = CACHE_DIR / "fct_match_candidates.parquet"
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        df["match_date"] = pd.to_datetime(df["match_date"])
        return df

    return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_teams() -> pd.DataFrame:
    """Load dimensional team ratings (Elo, rolling form, xG/xGA tiers)."""
    parquet_path = CACHE_DIR / "dim_teams.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_season_equity() -> pd.DataFrame:
    """Load 12 seasons of walk-forward betting equity and drawdown metrics."""
    parquet_path = CACHE_DIR / "rpt_season_equity.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_calibration_metrics() -> pd.DataFrame:
    """Load probabilistic scoring metrics (Brier, Log Loss, RPS, ECE) across all models."""
    parquet_path = CACHE_DIR / "rpt_calibration_metrics.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_reliability_curve() -> pd.DataFrame:
    """Load reliability curve bins (predicted vs observed frequency)."""
    parquet_path = CACHE_DIR / "rpt_reliability_curve.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_edge_profile() -> pd.DataFrame:
    """Load empirical ROI and hit rate by relative edge bucket."""
    parquet_path = CACHE_DIR / "rpt_edge_profile.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()


@st.cache_data(ttl=600, show_spinner=False)
def load_backtest_variants() -> pd.DataFrame:
    """Load comparative backtest matrix across models and execution types."""
    parquet_path = CACHE_DIR / "rpt_backtest_variants.parquet"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    return pd.DataFrame()
