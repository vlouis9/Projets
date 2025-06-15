#Stripped-Down MPG Auction Strategist (Clean Rewrite)

# Stripped-Down MPG Auction Strategist (Clean Rewrite)

import streamlit as st
import pandas as pd
import numpy as np
import re

--- Page Setup ---

st.set_page_config(page_title="MPG Strategist", layout="wide") st.title("⚽ MPG Auction Strategist — Clean Version")

--- Constants ---

KPI_KEYS = ["PerformanceEstimation", "PotentialEstimation", "RegularityEstimation", "GoalsEstimation"] CLUB_TIERS = {"Winner": 100, "European": 75, "Average": 50, "Relegation": 25} DEFAULT_WEIGHTS = { 'GK':  {"PerformanceEstimation": 0.5, "PotentialEstimation": 0.2, "RegularityEstimation": 0.3, "GoalsEstimation": 0.0}, 'DEF': {"PerformanceEstimation": 0.4, "PotentialEstimation": 0.2, "RegularityEstimation": 0.3, "GoalsEstimation": 0.1}, 'MID': {"PerformanceEstimation": 0.3, "PotentialEstimation": 0.2, "RegularityEstimation": 0.2, "GoalsEstimation": 0.3}, 'FWD': {"PerformanceEstimation": 0.3, "PotentialEstimation": 0.2, "RegularityEstimation": 0.1, "GoalsEstimation": 0.4}, }

--- Upload Section ---

st.markdown("### 📂 Upload Your Files") col1, col2 = st.columns(2) file_hist = col1.file_uploader("Historical MPG Data", type=["csv", "xls", "xlsx"]) file_new = col2.file_uploader("New Season File", type=["csv", "xls", "xlsx"])

@st.cache_data def load_file(f): return pd.read_excel(f) if f.name.endswith((".xls", ".xlsx")) else pd.read_csv(f)

@st.cache_data def simplify_position(pos): p = str(pos).upper().strip() return "GK" if p == "G" else "DEF" if p in ["D", "DL", "DC"] else "MID" if p in ["M", "MD", "MO"] else "FWD" if p == "A" else "UNKNOWN"

@st.cache_data def make_id(row): return f"{row['Joueur']}{simplify_position(row['Poste'])}{row['Club']}"

if file_hist and file_new: df_hist = load_file(file_hist) df_new = load_file(file_new)

for df in [df_hist, df_new]:
    df["simplified_position"] = df["Poste"].apply(simplify_position)
    df["player_id"] = df.apply(make_id, axis=1)

# --- Club Tier Section ---
st.markdown("### 🏆 Club Tier Assignment")
club_tier_input = {}
for c in sorted(df_new["Club"].dropna().unique()):
    tier = st.selectbox(f"{c}", list(CLUB_TIERS.keys()), index=2, key=f"tier_{c}")
    club_tier_input[c] = CLUB_TIERS[tier]
tier_weight = st.slider("Tier Weight in PVS", 0.0, 1.0, 0.25)

# --- Detect New & Returners ---
known_ids = set(df_hist['player_id'])
df_returning = df_new[df_new['player_id'].isin(known_ids)].copy()
df_new_only = df_new[~df_new['player_id'].isin(known_ids)].copy()

# --- Manual KPI Sliders ---
st.markdown("### ✍️ Subjective KPIs for New Players")
sliders = []
for _, row in df_new_only.iterrows():
    st.markdown(f"**{row['Joueur']} ({row['simplified_position']})**")
    entry = {"player_id": row['player_id'], "Joueur": row['Joueur'], "Poste": row['Poste'], "Club": row['Club'], "simplified_position": row['simplified_position']}
    for k in KPI_KEYS:
        entry[k] = st.slider(f"{k}", 0, 100, 50, key=row['player_id']+k)
    sliders.append(entry)
df_manual = pd.DataFrame(sliders)

for k in KPI_KEYS:
    df_manual[f"n_{k}"] = df_manual[k]

df_manual["club_tier"] = df_manual["Club"].map(club_tier_input).fillna(50)

def calc_pvs(row):
    pos = row["simplified_position"]
    weights = DEFAULT_WEIGHTS.get(pos, {})
    kpi_score = sum([row[f"n_{k}"] * weights.get(k, 0) for k in KPI_KEYS])
    return np.clip(kpi_score + row["club_tier"] * tier_weight, 0, 100)

df_manual["pvs"] = df_manual.apply(calc_pvs, axis=1)

# --- Display Final Table ---
st.markdown("### 🧮 Final Evaluated Players")
st.dataframe(df_manual[["Joueur", "Poste", "Club", "pvs"]].sort_values(by="pvs", ascending=False).reset_index(drop=True))

# --- Placeholder: Squad Builder ---
if st.button("💼 Build Squad"):
    st.info("You can plug in your select_squad() logic here with df_manual as input.")

else: st.warning("Upload both historical and new season files to proceed.")

