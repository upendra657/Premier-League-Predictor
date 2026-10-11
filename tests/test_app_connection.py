"""Tests for the Streamlit resilient connection and cache layer."""

from __future__ import annotations

import pandas as pd
import pytest

from app import connection as conn


def test_get_connection_info():
    info = conn.get_connection_info()
    assert info is not None
    assert info.mode in ("live", "snapshot")
    assert info.latency_ms >= 0


def test_load_bet_ledger():
    df = conn.load_bet_ledger()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "season" in df.columns
    assert "relative_edge" in df.columns
    assert "profit_gbp" in df.columns
    assert len(df) == 2154


def test_load_teams():
    df = conn.load_teams()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "team_name" in df.columns
    assert "elo" in df.columns
    assert "tier" in df.columns


def test_load_season_equity():
    df = conn.load_season_equity()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "Season" in df.columns
    assert "profit" in df.columns


def test_load_calibration_metrics():
    df = conn.load_calibration_metrics()
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "model" in df.columns
    assert "brier" in df.columns
