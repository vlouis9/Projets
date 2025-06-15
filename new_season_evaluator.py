import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import os
from typing import Dict, List, Tuple, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="MPG Combined Strategist",
    page_icon="⚽",
    layout="wide"
)

# --- Constants ---
DEFAULT_FORMATION = "4-4-2"
DEFAULT_SQUAD_SIZE = 20

# Club tiers and scores
CLUB_TIERS = ["Winner", "European", "Average", "Relegation"]
CLUB_TIER_SCORES = {
    "Winner": 100,
    "European": 75,
    "Average": 50,
    "Relegation": 25
}

# Subjective KPI keys for new players
SUBJECTIVE_KPI_KEYS = [
    "PerformanceEstimation",
    "PotentialEstimation",
    "RegularityEstimation",
    "GoalsEstimation"
]

# --- Cached Loader ---
@st.cache_data
def load_file(file):
    """Load Excel or CSV into DataFrame."""
    if file.name.lower().endswith(("xlsx", "xls")):
        return pd.read_excel(file)
    return pd.read_csv(file)

# --- Historical Logic (copied from your historical app) ---
class MPGAuctionStrategist:
    @staticmethod
    def simplify_position(pos: str) -> str:
        """Normalize positions to GK/DEF/MID/FWD."""
        p = str(pos).upper().strip()
        if p == "G":
            return "GK"
        if p in ["D", "DL", "DC"]:
            return "DEF"
        if p in ["M", "MD", "MO"]:
            return "MID"
        if p == "A":
            return "FWD"
        return "UNKNOWN"

    @staticmethod
    def create_player_id(row: pd.Series) -> str:
        """Unique player ID by name_position_club."""
        return f\"{row['Joueur']}_{MPGAuctionStrategist.simplify_position(row['Poste'])}_{row['Club']}\"

    @staticmethod
    def extract_rating_goals_starter(entry) -> Tuple[Optional[float], int, bool, bool]:
        """Parse a cell like '4.5**' or '(3.0)*' into (rating, goals, played, starter)."""
        if pd.isna(entry) or str(entry).strip() in ["", "0"]:
            return None, 0, False, False
        s = str(entry)
        goals = s.count("*")
        starter = "(" not in s
        try:
            rating = float(re.sub(r"[()*]", "", s))
            return rating, goals, True, starter
        except ValueError:
            return None, 0, False, False

    @staticmethod
    def get_gameweek_columns(cols: List[str]) -> List[str]:
        """Get columns matching D<number> sorted by number."""
        gw = [c for c in cols if re.fullmatch(r"D\\d+", c)]
        return sorted(gw, key=lambda x: int(x[1:]))

    @staticmethod
    def calculate_historical_kpis(df: pd.DataFrame, n_recent: int = 5) -> pd.DataFrame:
        """Compute season_avg, season_goals, reg_pct, recent_avg, recent_goals."""
        df = df.copy()
        gw_cols = MPGAuctionStrategist.get_gameweek_columns(df.columns)
        for col in ["season_avg", "season_goals", "reg_pct", "recent_avg", "recent_goals"]:
            df[col] = 0.0

        for i, row in df.iterrows():
            ratings, goals, starts = [], 0, 0
            # Season loop
            for gw in gw_cols:
                r, g, played, starter = MPGAuctionStrategist.extract_rating_goals_starter(row[gw])
                if played and r is not None:
                    ratings.append(r)
                    goals += g
                    starts += int(starter)
            df.at[i, "season_avg"] = np.mean(ratings) if ratings else 0.0
            df.at[i, "season_goals"] = goals
            df.at[i, "reg_pct"] = (starts / len(gw_cols) * 100) if gw_cols else 0.0
            # Recent loop
            rec = gw_cols[-n_recent:]
            rec_ratings, rec_goals = [], 0
            for gw in rec:
                r, g, played, _ = MPGAuctionStrategist.extract_rating_goals_starter(row[gw])
                if played and r is not None:
                    rec_ratings.append(r)
                    rec_goals += g
            df.at[i, "recent_avg"] = np.mean(rec_ratings) if rec_ratings else 0.0
            df.at[i, "recent_goals"] = rec_goals

        return df

    @staticmethod
    def normalize_kpis(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize to 0–100 scale."""
        df = df.copy()
        df["norm_season_avg"] = np.clip(df["season_avg"] * 10, 0, 100)
        df["norm_recent_avg"] = np.clip(df["recent_avg"] * 10, 0, 100)
        df["norm_reg"] = np.clip(df["reg_pct"], 0, 100)

        # Goals normalization per position
        for pos in ["DEF", "MID", "FWD"]:
            mask = df["simplified_position"] == pos
            df.loc[mask, "norm_recent_goals"] = np.clip(df.loc[mask, "recent_goals"] * 20, 0, 100)
            max_goals = df.loc[mask, "season_goals"].max() or 1
            df.loc[mask, "norm_season_goals"] = np.clip(
                df.loc[mask, "season_goals"] / max_goals * 100, 0, 100
            )

        df[["norm_recent_goals", "norm_season_goals"]] = df[["norm_recent_goals", "norm_season_goals"]].fillna(0)
        return df

    @staticmethod
    def calculate_pvs(
        df: pd.DataFrame,
        kpi_weights: Dict[str, Dict[str, float]],
        club_tiers: Dict[str, int],
        tier_weight: float,
    ) -> pd.DataFrame:
        """Combine all normalized KPIs + club tier into a single PVS."""
        df = df.copy()
        df["pvs"] = 0.0

        # KPI sum by position
        for pos, w in kpi_weights.items():
            mask = df["simplified_position"] == pos
            # Weighted sum
            df.loc[mask, "pvs"] = (
                df.loc[mask, "norm_recent_avg"] * w.get("recent_avg", 0)
                + df.loc[mask, "norm_season_avg"] * w.get("season_avg", 0)
                + df.loc[mask, "norm_reg"] * w.get("reg", 0)
                + df.loc[mask, "norm_recent_goals"] * w.get("recent_goals", 0)
                + df.loc[mask, "norm_season_goals"] * w.get("season_goals", 0)
            )

        # Add club tier impact
        if "Club" in df.columns:
            df["pvs"] += df["Club"].map(club_tiers).fillna(50) * tier_weight

        df["pvs"] = df["pvs"].clip(0, 100)
        return df

# --- Main App UI ---
def main():
    st.title("⚽ MPG Combined Auction Strategist")

    # Uploaders
    st.markdown("### 📂 Upload your files")
    col1, col2 = st.columns(2)
    hist_file = col1.file_uploader("Historical data (CSV/XLSX)", type=["csv", "xls", "xlsx"])
    new_file = col2.file_uploader("New season file (CSV/XLSX)", type=["csv", "xls", "xlsx"])

    if not hist_file or not new_file:
        st.info("Please upload both historical and new-season files to proceed.")
        return

    # Load data
    df_hist = load_file(hist_file)
    df_new = load_file(new_file)

    # Basic preprocessing
    for df in (df_hist, df_new):
        df["simplified_position"] = df["Poste"].apply(MPGAuctionStrategist.simplify_position)
        df["player_id"] = df.apply(MPGAuctionStrategist.create_player_id, axis=1)

    # Identify new vs returning
    known = set(df_hist["player_id"])
    all_new = set(df_new["player_id"])
    new_ids = all_new - known
    returning_ids = all_new & known
    df_returning = df_new[df_new["player_id"].isin(returning_ids)].copy()
    df_brand_new = df_new[df_new["player_id"].isin(new_ids)].copy()

    # Club tier assignment
    st.markdown("### 🏷️ Club Tier Assignment")
    club_tiers_ui: Dict[str, int] = {}
    for c in sorted(df_new["Club"].dropna().unique()):
        tier = st.selectbox(f"{c}", options=CLUB_TIERS, index=2, key=f"tier_{c}")
        club_tiers_ui[c] = CLUB_TIER_SCORES[tier]
    tier_weight = st.slider("Weight of Club Tier in PVS", min_value=0.0, max_value=1.0, value=0.25, step=0.05)

    # KPI weights for historical
    st.markdown("### ⚙️ KPI Weights (Historical)")
    # (Use your exact weights here; these are placeholders)
    kpi_weights = {
        "GK": {"recent_avg": 0.2, "season_avg": 0.5, "reg": 0.3},
        "DEF": {"recent_avg": 0.2, "season_avg": 0.3, "reg": 0.2, "recent_goals": 0.1, "season_goals": 0.2},
        "MID": {"recent_avg": 0.2, "season_avg": 0.2, "reg": 0.1, "recent_goals": 0.2, "season_goals": 0.3},
        "FWD": {"recent_avg": 0.2, "season_avg": 0.2, "reg": 0.1, "recent_goals": 0.2, "season_goals": 0.3},
    }

    # Compute historical PVS
    df_hist_kpis = MPGAuctionStrategist.calculate_historical_kpis(df_hist)
    df_hist_norm = MPGAuctionStrategist.normalize_kpis(df_hist_kpis)
    df_hist_pvs = MPGAuctionStrategist.calculate_pvs(df_hist_norm, kpi_weights, club_tiers_ui, tier_weight)

    # Subjective sliders for brand-new players
    st.markdown("### 🎚️ Subjective KPIs for New Players")
    manual_list = []
    for idx, row in df_brand_new.iterrows():
        st.subheader(f"{row['Joueur']} — {row['simplified_position']}")
        manual = {"player_id": row["player_id"], "Joueur": row["Joueur"], "Club": row["Club"], "Poste": row["simplified_position"]}
        for k in SUBJECTIVE_KPI_KEYS:
            manual[k] = st.slider(f"{k}", 0, 100, 50, key=f"{row['player_id']}_{k}")
        manual_list.append(manual)

    # Build DataFrame for manual entries
    if manual_list:
        df_manual = pd.DataFrame(manual_list)
        # Normalize
        for k in SUBJECTIVE_KPI_KEYS:
            df_manual[f"norm_{k}"] = df_manual[k].clip(0, 100)
        # Compute PVS for new players (equal weighting + club tier)
        df_manual["pvs"] = df_manual[[f"norm_{k}" for k in SUBJECTIVE_KPI_KEYS]].mean(axis=1)
        df_manual["pvs"] += df_manual["Club"].map(club_tiers_ui).fillna(50) * tier_weight
        df_manual["pvs"] = df_manual["pvs"].clip(0, 100)
    else:
        df_manual = pd.DataFrame(columns=list(df_hist_pvs.columns) + ["pvs"])

    # Combine all evaluated players
    df_all = pd.concat([df_hist_pvs, df_manual], ignore_index=True, sort=False)
    st.markdown("### 📊 All Players with PVS")
    st.dataframe(df_all[["Joueur", "Poste", "Club", "pvs"]].sort_values("pvs", ascending=False).reset_index(drop=True))

    # Placeholder: Squad building
    if st.button("🔨 Build Squad"):
        # You can call your historical select_squad method here:
        # strategist = MPGAuctionStrategist()
        # squad_df, summary = strategist.select_squad(df_all, DEFAULT_FORMATION, DEFAULT_SQUAD_SIZE, min_recent_games=1)
        # st.write(summary)
        # st.dataframe(squad_df)
        st.info("Squad builder not yet wired. Paste your select_squad() call here.")

if __name__ == "__main__":
    main()
