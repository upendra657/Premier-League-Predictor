"""Generate certified Apache Parquet cache snapshots and SQLite analytical tables.

Extracts reporting datasets from reports/ and artifacts/ and outputs them to
app/cache/*.parquet and an indexed SQLite database for dual-mode access.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
CACHE_DIR = ROOT / "app" / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


def generate_cache() -> None:
    print(f"Generating certified reporting cache in {CACHE_DIR}...")

    # 1. Bet Ledger
    excel_path = ROOT / "reports" / "epl_decision_engine_results.xlsx"
    if excel_path.exists():
        df_bets = pd.read_excel(excel_path, sheet_name="Bet Ledger")
        df_bets.columns = [
            c.lower().replace(" ", "_").replace("_(£)", "_gbp").replace("(", "").replace(")", "")
            for c in df_bets.columns
        ]
        df_bets["match_date"] = pd.to_datetime(df_bets["match_date"])
        df_bets["won"] = df_bets["won"].astype(int)
        df_bets["selection_desc"] = df_bets["selection"].map(
            {"H": "Home Win", "D": "Draw", "A": "Away Win"}
        ).fillna(df_bets["selection"])
    else:
        # Fallback if excel is missing
        results = json.loads((ROOT / "reports" / "results.json").read_text())
        df_bets = pd.DataFrame(results.get("equity_by_season", []))

    bets_parquet = CACHE_DIR / "fct_bet_ledger.parquet"
    df_bets.to_parquet(bets_parquet, index=False)
    print(f"  -> {bets_parquet.name}: {len(df_bets):,} rows")

    # 2. All Candidates
    if excel_path.exists():
        df_cand = pd.read_excel(excel_path, sheet_name="All Candidates")
        df_cand.columns = [
            c.lower().replace(" ", "_").replace("_(£)", "_gbp").replace("(", "").replace(")", "")
            for c in df_cand.columns
        ]
        df_cand["match_date"] = pd.to_datetime(df_cand["match_date"])
        df_cand["won"] = pd.to_numeric(df_cand["won"], errors="coerce").fillna(0).astype(int)
        df_cand["selected"] = pd.to_numeric(df_cand["selected"], errors="coerce").fillna(0).astype(int)
    else:
        df_cand = pd.DataFrame()

    cand_parquet = CACHE_DIR / "fct_match_candidates.parquet"
    df_cand.to_parquet(cand_parquet, index=False)
    print(f"  -> {cand_parquet.name}: {len(df_cand):,} rows")

    # 3. Dim Teams (Elo, Form, xG, xGA)
    teams_json = ROOT / "artifacts" / "team_snapshot.json"
    if teams_json.exists():
        raw_teams = json.loads(teams_json.read_text())
        records = []
        for team, d in raw_teams.items():
            records.append({
                "team_name": team,
                "as_of": d.get("as_of"),
                "elo": float(d.get("elo", 1500.0)),
                "points_roll": float(d.get("points_roll", 1.35)),
                "xg_roll": float(d.get("xg_roll", 1.35)),
                "xga_roll": float(d.get("xga_roll", 1.35)),
                "goal_diff_roll": float(d.get("xg_roll", 1.35)) - float(d.get("xga_roll", 1.35)),
            })
        df_teams = pd.DataFrame(records).sort_values("elo", ascending=False).reset_index(drop=True)
        df_teams["rank"] = df_teams.index + 1
        
        # Tier assignment
        def assign_tier(rank: int) -> str:
            if rank <= 4:
                return "Tier 1: Elite / UCL"
            elif rank <= 8:
                return "Tier 2: European Contenders"
            elif rank <= 14:
                return "Tier 3: Mid-Table Stable"
            return "Tier 4: Relegation Risk"
        
        df_teams["tier"] = df_teams["rank"].apply(assign_tier)
    else:
        df_teams = pd.DataFrame()

    teams_parquet = CACHE_DIR / "dim_teams.parquet"
    df_teams.to_parquet(teams_parquet, index=False)
    print(f"  -> {teams_parquet.name}: {len(df_teams):,} rows")

    # 4. Season Equity & Performance
    results = json.loads((ROOT / "reports" / "results.json").read_text())
    equity_list = results.get("equity_by_season", [])
    df_equity = pd.DataFrame(equity_list)
    df_equity["win_loss_streak"] = np.where(df_equity["profit"] > 0, "Profitable", "Drawdown")
    equity_parquet = CACHE_DIR / "rpt_season_equity.parquet"
    df_equity.to_parquet(equity_parquet, index=False)
    print(f"  -> {equity_parquet.name}: {len(df_equity):,} rows")

    # 5. Model Calibration Metrics
    metrics_list = results.get("walk_forward_metrics", [])
    df_metrics = pd.DataFrame(metrics_list)
    metrics_parquet = CACHE_DIR / "rpt_calibration_metrics.parquet"
    df_metrics.to_parquet(metrics_parquet, index=False)
    print(f"  -> {metrics_parquet.name}: {len(df_metrics):,} rows")

    # 6. Reliability Curve Bins
    rel_curve_list = results.get("reliability_curve", [])
    df_rel = pd.DataFrame(rel_curve_list)
    rel_parquet = CACHE_DIR / "rpt_reliability_curve.parquet"
    df_rel.to_parquet(rel_parquet, index=False)
    print(f"  -> {rel_parquet.name}: {len(df_rel):,} rows")

    # 7. Edge Profile Bins
    edge_profile_list = results.get("edge_profile", [])
    df_edge = pd.DataFrame(edge_profile_list)
    edge_parquet = CACHE_DIR / "rpt_edge_profile.parquet"
    df_edge.to_parquet(edge_parquet, index=False)
    print(f"  -> {edge_parquet.name}: {len(df_edge):,} rows")

    # 8. Strategy Backtest Variants Comparison
    backtest_csv = ROOT / "reports" / "backtest_comparison.csv"
    if backtest_csv.exists():
        df_variants = pd.read_csv(backtest_csv)
    else:
        df_variants = pd.DataFrame()
    variants_parquet = CACHE_DIR / "rpt_backtest_variants.parquet"
    df_variants.to_parquet(variants_parquet, index=False)
    print(f"  -> {variants_parquet.name}: {len(df_variants):,} rows")

    # 9. Also populate the local SQLite Database for live mode demonstration
    db_path = CACHE_DIR / "epl_analytics.db"
    print(f"Building local SQLite database at {db_path}...")
    with sqlite3.connect(db_path) as conn:
        df_bets.to_sql("fct_bet_ledger", conn, if_exists="replace", index=False)
        df_cand.to_sql("fct_match_candidates", conn, if_exists="replace", index=False)
        df_teams.to_sql("dim_teams", conn, if_exists="replace", index=False)
        df_equity.to_sql("rpt_season_equity", conn, if_exists="replace", index=False)
        df_metrics.to_sql("rpt_calibration_metrics", conn, if_exists="replace", index=False)
        df_rel.to_sql("rpt_reliability_curve", conn, if_exists="replace", index=False)
        df_edge.to_sql("rpt_edge_profile", conn, if_exists="replace", index=False)
        df_variants.to_sql("rpt_backtest_variants", conn, if_exists="replace", index=False)

        # Build indexes for sub-second query performance
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bets_season ON fct_bet_ledger (season);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_bets_teams ON fct_bet_ledger (home_team, away_team);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_cand_season ON fct_match_candidates (season);")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_teams_name ON dim_teams (team_name);")
    print(f"  -> Successfully populated SQLite database: {db_path.name}")
    print("Snapshot Cache Generation Complete!")


if __name__ == "__main__":
    generate_cache()
