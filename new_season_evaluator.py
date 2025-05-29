import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, Tuple

# ───────────────────────────────────────────────────────────────
# 0. Page Configuration (must be first)
# ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MPG Auction Strategist – New Season Mode",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 New Season Mode: Composite Player Evaluation")

# ───────────────────────────────────────────────────────────────
# 1. Default Profile & Session State Initialization
# ───────────────────────────────────────────────────────────────
DEFAULT_PROFILE = {
    "weights": {
        "base": 1.0,
        "talent": 1.0,
        "buzz": 1.0,
        "expert": 1.0,
    },
    "mrb_params": {
        "GK": {"max_proportional_bonus_at_pvs100": 0.3},
        "DEF": {"max_proportional_bonus_at_pvs100": 0.4},
        "MID": {"max_proportional_bonus_at_pvs100": 0.6},
        "FWD": {"max_proportional_bonus_at_pvs100": 0.8},
    },
    "formation": "4-4-2",
    "squad_size": 20,
    "min_recent_filter": 1,
}

# Initialize session state for weights and squad parameters
for key, value in DEFAULT_PROFILE["weights"].items():
    if key not in st.session_state:
        st.session_state[key] = value
if "mrb_params" not in st.session_state:
    st.session_state.mrb_params = DEFAULT_PROFILE["mrb_params"].copy()
if "formation_key" not in st.session_state:
    st.session_state.formation_key = DEFAULT_PROFILE["formation"]
if "squad_size" not in st.session_state:
    st.session_state.squad_size = DEFAULT_PROFILE["squad_size"]
if "min_recent_filter" not in st.session_state:
    st.session_state.min_recent_filter = DEFAULT_PROFILE["min_recent_filter"]

# ───────────────────────────────────────────────────────────────
# 2. File Upload & Data Loading (Cached)
# ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data(uploaded_file) -> pd.DataFrame:
    if uploaded_file.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded_file)
    return pd.read_excel(uploaded_file)

uploaded_file = st.sidebar.file_uploader(
    "📁 Upload New Season Data (CSV/Excel)",
    type=["csv", "xlsx", "xls"],
)
if not uploaded_file:
    st.info("Please upload a data file to begin.")
    st.stop()

df_raw = load_data(uploaded_file)
st.subheader("1️⃣ Raw Data Preview")
st.dataframe(df_raw.head(), use_container_width=True)

# ───────────────────────────────────────────────────────────────
# 3. Auto‐Fill Missing Parameters
# ───────────────────────────────────────────────────────────────
def fill_missing_params(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    n = len(df2)
    if "talent_potential" not in df2.columns:
        df2["talent_potential"] = np.random.uniform(4, 8, n).round(1)
    if "market_buzz" not in df2.columns:
        df2["market_buzz"] = np.random.uniform(3, 9, n).round(1)
    if "expert_sentiment" not in df2.columns:
        df2["expert_sentiment"] = np.random.uniform(5, 9, n).round(1)
    return df2

df_filled = fill_missing_params(df_raw)

# ───────────────────────────────────────────────────────────────
# 4. Normalize Player Attributes (Cached)
# ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def normalize_player_attributes(df: pd.DataFrame) -> pd.DataFrame:
    dfn = df.copy()
    # Convert "Cote" to numeric (ensuring a value between 1 and 100)
    dfn["Cote"] = pd.to_numeric(dfn["Cote"], errors="coerce").fillna(1).clip(1, 100).astype(int)
    dfn["norm_cote"] = dfn["Cote"]
    # Assume scouting parameters are on a 0–10 scale; scale to 0–100.
    dfn["norm_talent"] = dfn["talent_potential"].astype(float).clip(0, 10) * 10
    dfn["norm_buzz"]   = dfn["market_buzz"].astype(float).clip(0, 10) * 10
    dfn["norm_expert"] = dfn["expert_sentiment"].astype(float).clip(0, 10) * 10
    return dfn

df_norm = normalize_player_attributes(df_filled)

# ───────────────────────────────────────────────────────────────
# 5. Sidebar Settings: Composite Score Weights & MRB Parameters + Reset
# ───────────────────────────────────────────────────────────────
with st.sidebar.expander("🎯 Composite Score Settings", expanded=True):
    st.session_state["base"] = st.slider("Cote Weight", 0.1, 5.0, st.session_state["base"], 0.1,
                                           help="Weight for the player's current rating (Cote).")
    st.session_state["talent"] = st.slider("Talent Potential Weight", 0.1, 5.0, st.session_state["talent"], 0.1,
                                             help="Weight for talent (0–10 scale).")
    st.session_state["buzz"] = st.slider("Market Buzz Weight", 0.1, 5.0, st.session_state["buzz"], 0.1,
                                           help="Weight for buzz (0–10 scale).")
    st.session_state["expert"] = st.slider("Expert Sentiment Weight", 0.1, 5.0, st.session_state["expert"], 0.1,
                                             help="Weight for expert opinions (0–10 scale).")

with st.sidebar.expander("💰 MRB Parameters"):
    for pos in ["GK", "DEF", "MID", "FWD"]:
        st.session_state.mrb_params[pos]["max_proportional_bonus_at_pvs100"] = st.slider(
            f"{pos} Bonus @ PVS=100", 0.0, 1.0,
            st.session_state.mrb_params[pos]["max_proportional_bonus_at_pvs100"],
            0.01,
            help="Proportional bonus applied on Cote if PVS=100 (capped at 2×Cote)."
        )

with st.sidebar.expander("🔄 Controls"):
    if st.button("Reset to Default"):
        st.session_state.clear()
        st.experimental_rerun()

# ───────────────────────────────────────────────────────────────
# 6. Calculate Composite Player Value Score (PVS) (Cached)
# ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def calculate_pvs(df: pd.DataFrame, w: Dict[str, float]) -> pd.DataFrame:
    d = df.copy()
    total_w = w["base"] + w["talent"] + w["buzz"] + w["expert"]
    d["pvs"] = (
        d["norm_cote"] * w["base"] +
        d["norm_talent"] * w["talent"] +
        d["norm_buzz"]   * w["buzz"] +
        d["norm_expert"] * w["expert"]
    ) / max(total_w, 1e-6)
    d["pvs"] = d["pvs"].clip(0, 100)
    return d

df_pvs = calculate_pvs(df_norm, {
    "base": st.session_state["base"],
    "talent": st.session_state["talent"],
    "buzz": st.session_state["buzz"],
    "expert": st.session_state["expert"],
})

# ───────────────────────────────────────────────────────────────
# 7. Calculate MRB & Value-per-Cost (Cached)
# ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def calculate_mrb(df: pd.DataFrame, mrb_params: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    d = df.copy()
    d["mrb"] = d["Cote"].astype(int)  # Base allocation from Cote
    # Expect that a column "simplified_position" exists; if not, we create a simple version
    if "simplified_position" not in d.columns:
        # For demo purposes, assume positions are identified in a 'Poste' column
        d["simplified_position"] = d["Poste"].fillna("").apply(
            lambda p: "GK" if p.strip().upper() in ["G", "GK"] 
            else ("DEF" if p.strip().upper() in ["D", "DEF"] 
                  else ("MID" if p.strip().upper() in ["M", "MID"] else "FWD"))
        )
    for pos, params in mrb_params.items():
        mask = d["simplified_position"] == pos
        if mask.any():
            m = params["max_proportional_bonus_at_pvs100"]
            def _calc(row):
                base = row["Cote"]
                bonus = (row["pvs"] / 100.0) * m
                val = base * (1 + bonus)
                return int(round(min(val, base * 2)))
            d.loc[mask, "mrb"] = d.loc[mask].apply(_calc, axis=1)
    d["value_per_cost"] = d["pvs"] / d["mrb"].replace(0, np.nan)
    d["value_per_cost"].fillna(0, inplace=True)
    return d

df_eval = calculate_mrb(df_pvs, st.session_state.mrb_params)

# ───────────────────────────────────────────────────────────────
# 8. Searchable Evaluated Players Table & Download
# ───────────────────────────────────────────────────────────────
st.subheader("2️⃣ Evaluated Players")
search_query = st.text_input("🔍 Search Evaluated Players")
df_show = df_eval.copy()
if search_query:
    df_show = df_show[df_show.apply(lambda r: r.astype(str).str.contains(search_query, case=False).any(), axis=1)]
st.dataframe(df_show.reset_index(drop=True), use_container_width=True, height=300)
st.download_button("📥 Download Evaluated Data CSV",
                   data=df_eval.to_csv(index=False).encode("utf-8"),
                   file_name="evaluated_players.csv",
                   mime="text/csv")

# ───────────────────────────────────────────────────────────────
# 9. Squad Building via Historical Logic (Self-Contained Version)
# ───────────────────────────────────────────────────────────────
# Here we include a simplified version of your historical squad-selection process.
class MPGAuctionStrategist:
    def __init__(self):
        self.formations = {
            "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
            "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
            "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
            "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
            "4-5-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
            "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},
            "5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1},
        }
        self.squad_minimums = {"GK": 2, "DEF": 6, "MID": 6, "FWD": 4}

    @property
    def squad_minimums_sum_val(self) -> int:
        return sum(self.squad_minimums.values())

    def select_squad(self, df: pd.DataFrame, formation_key: str, target_squad_size: int, min_recent: int) -> Tuple[pd.DataFrame, Dict]:
        # For demonstration, we perform a basic selection:
        # 1. Filter players by a column "recent_games_played_count" if present
        d = df.copy()
        if "recent_games_played_count" in d.columns:
            d = d[d["recent_games_played_count"] >= min_recent]
        # 2. Sort by descending PVS
        d = d.sort_values(by="pvs", ascending=False)
        # 3. Select the top 'target_squad_size' players
        squad_df = d.head(target_squad_size).copy()
        summary = {
            "total_players": len(squad_df),
            "total_pvs": float(squad_df["pvs"].sum()),
        }
        return squad_df, summary

# Sidebar – Squad Building Settings
st.sidebar.markdown("---")
st.sidebar.markdown("🏆 Squad Building Settings")
# Formation
strategist = MPGAuctionStrategist()
formation = st.sidebar.selectbox("Select Formation",
                                 options=list(strategist.formations.keys()),
                                 index=list(strategist.formations.keys()).index(st.session_state.formation_key))
# Squad Size
squad_size = st.sidebar.number_input("Squad Size",
                                     min_value=strategist.squad_minimums_sum_val,
                                     max_value=30,
                                     value=st.session_state.squad_size)
# Minimum required recent games
min_recent_games = st.sidebar.number_input("Min Recent Games Played",
                                           min_value=0,
                                           max_value=10,
                                           value=st.session_state.min_recent_filter)

squad_df, squad_summary = strategist.select_squad(df_eval, formation, squad_size, min_recent_games)

st.subheader("🏅 Suggested Squad")
st.dataframe(squad_df.reset_index(drop=True), use_container_width=True, height=300)
st.subheader("📈 Squad Summary")
st.write(squad_summary)
st.download_button("📥 Download Squad CSV",
                   data=squad_df.to_csv(index=False).encode("utf-8"),
                   file_name="suggested_squad.csv",
                   mime="text/csv")
