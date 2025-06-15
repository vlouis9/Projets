import streamlit as st
import pandas as pd
import numpy as np
import re
import json
import os
from typing import Dict, List, Tuple, Optional

--- Page Configuration ---

st.set_page_config(page_title="MPG Combined Strategist", page_icon="⚽", layout="wide")

--- Constants ---

DEFAULT_FORMATION = "4-4-2" DEFAULT_SQUAD_SIZE = 20 CLUB_TIER_SCORES = {"Winner": 100, "European": 75, "Average": 50, "Relegation": 25} SUBJECTIVE_KPI_KEYS = ["PerformanceEstimation", "PotentialEstimation", "RegularityEstimation", "GoalsEstimation"]

--- Class Reuse from Historical App ---

from io import BytesIO @st.cache_data def load_excel_or_csv(file): return pd.read_excel(file) if file.name.endswith(('xls', 'xlsx')) else pd.read_csv(file)

class MPGAuctionStrategist: @staticmethod def simplify_position(pos): p = str(pos).upper().strip() return 'GK' if p == 'G' else 'DEF' if p in ['D', 'DL', 'DC'] else 'MID' if p in ['M', 'MD', 'MO'] else 'FWD' if p == 'A' else 'UNKNOWN'

@staticmethod
def create_player_id(row):
    return f"{row['Joueur']}_{MPGAuctionStrategist.simplify_position(row['Poste'])}_{row['Club']}"

@staticmethod
def extract_rating_goals_starter(rating_str):
    if pd.isna(rating_str) or str(rating_str).strip() in ['', '0']:
        return None, 0, False, False
    s = str(rating_str)
    goals = s.count('*')
    starter = '(' not in s
    try:
        rating = float(re.sub(r'[()*]', '', s))
        return rating, goals, True, starter
    except:
        return None, 0, False, False

@staticmethod
def get_gameweek_columns(columns):
    return sorted([c for c in columns if re.fullmatch(r'D\d+', c)], key=lambda x: int(x[1:]))

@staticmethod
def calculate_historical_kpis(df, n_recent=5):
    df = df.copy()
    gw_cols = MPGAuctionStrategist.get_gameweek_columns(df.columns)
    for col in ['season_avg', 'season_goals', 'reg_pct', 'recent_avg', 'recent_goals']:
        df[col] = 0.0
    for i, row in df.iterrows():
        ratings, goals, starts = [], 0, 0
        for gw in gw_cols:
            r, g, played, starter = MPGAuctionStrategist.extract_rating_goals_starter(row[gw])
            if played and r is not None:
                ratings.append(r); goals += g; starts += int(starter)
        df.at[i, 'season_avg'] = np.mean(ratings) if ratings else 0
        df.at[i, 'season_goals'] = goals
        df.at[i, 'reg_pct'] = (starts / len(gw_cols) * 100) if gw_cols else 0
        # recent
        rec_ratings, rec_goals = [], 0
        for gw in gw_cols[-n_recent:]:
            r, g, played, _ = MPGAuctionStrategist.extract_rating_goals_starter(row[gw])
            if played and r is not None:
                rec_ratings.append(r); rec_goals += g
        df.at[i, 'recent_avg'] = np.mean(rec_ratings) if rec_ratings else 0
        df.at[i, 'recent_goals'] = rec_goals
    return df

@staticmethod
def normalize_kpis(df):
    df = df.copy()
    df['norm_season_avg'] = np.clip(df['season_avg'] * 10, 0, 100)
    df['norm_recent_avg'] = np.clip(df['recent_avg'] * 10, 0, 100)
    df['norm_reg'] = np.clip(df['reg_pct'], 0, 100)
    for pos in ['DEF', 'MID', 'FWD']:
        m = df['simplified_position'] == pos
        df.loc[m, 'norm_recent_goals'] = np.clip(df.loc[m, 'recent_goals'] * 20, 0, 100)
        max_goals = df.loc[m, 'season_goals'].max() or 1
        df.loc[m, 'norm_season_goals'] = np.clip(df.loc[m, 'season_goals'] / max_goals * 100, 0, 100)
    df['norm_recent_goals'].fillna(0, inplace=True)
    df['norm_season_goals'].fillna(0, inplace=True)
    return df

@staticmethod
def calculate_pvs(df, weights, club_tiers, tier_weight):
    df = df.copy(); df['pvs'] = 0.0
    for pos, w in weights.items():
        m = df['simplified_position'] == pos
        df.loc[m, 'pvs'] = (
            df.loc[m, 'norm_recent_avg'] * w.get('recent_avg', 0) +
            df.loc[m, 'norm_season_avg'] * w.get('season_avg', 0) +
            df.loc[m, 'norm_reg'] * w.get('reg', 0) +
            df.loc[m, 'norm_recent_goals'] * w.get('recent_goals', 0) +
            df.loc[m, 'norm_season_goals'] * w.get('season_goals', 0)
        )
        if 'Club' in df.columns:
            df.loc[m, 'pvs'] += df.loc[m, 'Club'].map(club_tiers).fillna(50) * tier_weight
    df['pvs'] = df['pvs'].clip(0, 100)
    return df

--- UI App ---

st.title("⚽ MPG Combined Auction Strategist")

st.markdown("### 📁 Upload Files") col1, col2 = st.columns(2) hist_file = col1.file_uploader("Upload historical data file", type=['csv', 'xls', 'xlsx']) new_file = col2.file_uploader("Upload new season file", type=['csv', 'xls', 'xlsx'])

if hist_file and new_file: df_hist = load_excel_or_csv(hist_file) df_new = load_excel_or_csv(new_file)

for df in [df_hist, df_new]:
    df['simplified_position'] = df['Poste'].apply(MPGAuctionStrategist.simplify_position)
    df['player_id'] = df.apply(MPGAuctionStrategist.create_player_id, axis=1)

# Merge logic
known_ids = set(df_hist['player_id'])
new_ids = set(df_new['player_id'])
new_players_df = df_new[df_new['player_id'].isin(new_ids - known_ids)].copy()
returning_players_df = df_new[df_new['player_id'].isin(new_ids & known_ids)].copy()

# Club tier editor
st.markdown("### 🏷 Club Tier Assignment")
unique_clubs = sorted(df_new['Club'].dropna().unique())
club_tiers_ui = {}
for club in unique_clubs:
    tier = st.selectbox(f"{club}", options=list(CLUB_TIER_SCORES.keys()), index=2, key=f"tier_{club}")
    club_tiers_ui[club] = CLUB_TIER_SCORES[tier]
tier_weight = st.slider("⚖️ Weight of Club Tier in PVS", 0.0, 1.0, 0.25, step=0.05)

# Historical KPI weights
st.markdown("### ⚙️ KPI Weights")
default_weights = {
    'GK':  {'recent_avg': 0.2, 'season_avg': 0.5, 'reg': 0.3},
    'DEF': {'recent_avg': 0.2, 'season_avg': 0.3, 'reg': 0.2, 'recent_goals': 0.1, 'season_goals': 0.2},
    'MID': {'recent_avg': 0.2, 'season_avg': 0.2, 'reg': 0.1, 'recent_goals': 0.2, 'season_goals': 0.3},
    'FWD': {'recent_avg': 0.2, 'season_avg': 0.2, 'reg': 0.1, 'recent_goals': 0.2, 'season_goals': 0.3}
}

# Historical KPI Calculation
df_hist_kpis = MPGAuctionStrategist.calculate_historical_kpis(df_hist)
df_hist_norm = MPGAuctionStrategist.normalize_kpis(df_hist_kpis)
df_hist_with_pvs = MPGAuctionStrategist.calculate_pvs(df_hist_norm, default_weights, club_tiers_ui, tier_weight)

# Normalize new player sliders
st.markdown("### 🎚 Subjective KPI Assignment for New Players")
final_slider_data = []
for _, row in new_players_df.iterrows():
    st.subheader(row['Joueur'])
    pos = row['simplified_position']
    kpi_row = {'player_id': row['player_id'], 'Joueur': row['Joueur'], 'Club': row['Club'], 'Poste': pos}
    for kpi in SUBJECTIVE_KPI_KEYS:
        val = st.slider(f"{kpi} ({pos})", 0, 100, 50, key=row['player_id']+kpi)
        kpi_row[kpi] = val
    final_slider_data.append(kpi_row)

if final_slider_data:
    df_manual = pd.DataFrame(final_slider_data)
    for k in SUBJECTIVE_KPI_KEYS:
        df_manual[f'n_{k}'] = df_manual[k].clip(0, 100)
    df_manual['pvs'] = df_manual[[f'n_{k}' for k in SUBJECTIVE_KPI_KEYS]].mean(axis=1) + df_manual['Club'].map(club_tiers_ui).fillna(50) * tier_weight
    df_manual['pvs'] = df_manual['pvs'].clip(0, 100)

    # Merge all players
    full_df = pd.concat([df_hist_with_pvs, df_manual], ignore_index=True, sort=False)
    st.markdown("### 🧮 Final Player Table")
    st.dataframe(full_df[['Joueur','Poste','Club','pvs']].sort_values(by='pvs', ascending=False).reset_index(drop=True))

else: st.info("Please upload both historical and new season files to begin.")

