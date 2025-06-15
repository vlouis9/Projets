import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import os
from typing import Dict, List, Tuple, Optional, Set

# --- Page Configuration ---
st.set_page_config(
    page_title="MPG Combined Strategist",
    page_icon="⚽",
    layout="wide"
)

# --- Constants ---
DEFAULT_FORMATION = "4-4-2"
DEFAULT_SQUAD_SIZE = 20
CLUB_TIER_SCORES = {"Winner": 100, "European": 75, "Average": 50, "Relegation": 25}
SUBJECTIVE_KPI_KEYS = ["PerformanceEstimation", "PotentialEstimation", "RegularityEstimation", "GoalsEstimation"]

# --- Load File ---
@st.cache_data
def load_excel_or_csv(file):
    return pd.read_excel(file) if file.name.endswith(('xls', 'xlsx')) else pd.read_csv(file)

# --- UI ---
st.title("⚽ MPG Combined Strategist")
st.markdown("Upload both historical and new season files to begin.")

col1, col2 = st.columns(2)
hist_file = col1.file_uploader("📂 Historical data file", type=["csv", "xls", "xlsx"])
new_file = col2.file_uploader("📂 New season file", type=["csv", "xls", "xlsx"])

if hist_file and new_file:
    df_hist = load_excel_or_csv(hist_file)
    df_new = load_excel_or_csv(new_file)

    for df in [df_hist, df_new]:
        df['simplified_position'] = df['Poste'].apply(MPGAuctionStrategist.simplify_position)
        df['player_id'] = df.apply(MPGAuctionStrategist.create_player_id, axis=1)

    # --- Identify Players ---
    known_ids = set(df_hist['player_id'])
    new_ids = set(df_new['player_id'])

    new_players_df = df_new[df_new['player_id'].isin(new_ids - known_ids)].copy()
    returning_players_df = df_new[df_new['player_id'].isin(new_ids & known_ids)].copy()
    removed_players_df = df_hist[~df_hist['player_id'].isin(new_ids)].copy()

    st.markdown(f"**🆕 New players detected**: {len(new_players_df)}")
    st.markdown(f"**👋 Removed players**: {len(removed_players_df)}")

    # --- Club Tier Editor ---
    st.markdown("### 🏆 Club Tiers")
    club_tiers = {}
    for club in sorted(df_new['Club'].dropna().unique()):
        tier = st.selectbox(f"{club}", options=list(CLUB_TIER_SCORES.keys()), index=2, key=f"tier_{club}")
        club_tiers[club] = CLUB_TIER_SCORES[tier]
    tier_weight = st.slider("Weight of Club Tier in PVS", 0.0, 1.0, 0.25, step=0.05)

    # --- Normalize New Player Slider Ranges ---
    df_hist_kpis = MPGAuctionStrategist.calculate_historical_kpis(df_hist)
    df_hist_norm = MPGAuctionStrategist.normalize_kpis(df_hist_kpis)

    kpi_ranges = {}
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        sub = df_hist_norm[df_hist_norm['simplified_position'] == pos]
        kpi_ranges[pos] = {}
        for kpi in SUBJECTIVE_KPI_KEYS:
            col = f"norm_{kpi.lower()}" if f"norm_{kpi.lower()}" in sub.columns else None
            kpi_ranges[pos][kpi] = (0, 100)  # default fallback

    # --- Manual Slider Input ---
    st.markdown("### 🎛️ Manual KPI sliders for new players")
    manual_data = []
    for _, row in new_players_df.iterrows():
        st.markdown(f"**{row['Joueur']} ({row['simplified_position']})**")
        sliders = {}
        for kpi in SUBJECTIVE_KPI_KEYS:
            val = st.slider(
                f"{kpi}", min_value=0, max_value=100, value=50,
                key=f"{row['player_id']}_{kpi}"
            )
            sliders[kpi] = val
        sliders.update({
            "player_id": row['player_id'],
            "Joueur": row['Joueur'],
            "Poste": row['Poste'],
            "Club": row['Club'],
            "simplified_position": row['simplified_position'],
        })
        manual_data.append(sliders)

    df_manual = pd.DataFrame(manual_data)
    for k in SUBJECTIVE_KPI_KEYS:
        df_manual[f'norm_{k}'] = df_manual[k].clip(0, 100)
    df_manual['pvs'] = df_manual[[f'norm_{k}' for k in SUBJECTIVE_KPI_KEYS]].mean(axis=1)
    df_manual['pvs'] += df_manual['Club'].map(club_tiers).fillna(50) * tier_weight
    df_manual['pvs'] = df_manual['pvs'].clip(0, 100)

    # --- Evaluate Returning Players ---
    default_weights = {
        'GK': {'recent_avg': 0.2, 'season_avg': 0.5, 'reg': 0.3},
        'DEF': {'recent_avg': 0.2, 'season_avg': 0.3, 'reg': 0.2, 'recent_goals': 0.1, 'season_goals': 0.2},
        'MID': {'recent_avg': 0.2, 'season_avg': 0.2, 'reg': 0.1, 'recent_goals': 0.2, 'season_goals': 0.3},
        'FWD': {'recent_avg': 0.2, 'season_avg': 0.2, 'reg': 0.1, 'recent_goals': 0.2, 'season_goals': 0.3}
    }

    df_returning_kpis = df_hist_norm[df_hist_norm['player_id'].isin(returning_players_df['player_id'])].copy()
    df_returning_eval = MPGAuctionStrategist.calculate_pvs(df_returning_kpis, default_weights, club_tiers, tier_weight)

    # --- Combine All Evaluated Players ---
    df_all_eval = pd.concat([df_returning_eval, df_manual], ignore_index=True, sort=False)

    # --- Display Final Player Table ---
    st.markdown("### 🧮 Final Evaluated Players")
    st.dataframe(df_all_eval[['Joueur', 'Poste', 'Club', 'pvs']].sort_values(by='pvs', ascending=False).reset_index(drop=True))

    # --- Squad Builder Button (uses existing logic) ---
    if st.button("💼 Build Squad with select_squad"):
        strategist = MPGAuctionStrategist()
        squad_df, summary = strategist.select_squad(df_all_eval, DEFAULT_FORMATION, DEFAULT_SQUAD_SIZE)
        st.success("Squad successfully built!")
        st.dataframe(squad_df)
        st.json(summary)
else:
    st.info("Please upload both historical and new season files to begin.")
