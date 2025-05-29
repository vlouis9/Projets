# new_season_app.py

import streamlit as st
import pandas as pd
import numpy as np
from typing import Dict
from season_with_data import MPGAuctionStrategist

# =============================================================================
# 0. Page Configuration
# =============================================================================
st.set_page_config(
    page_title="MPG Auction Strategist – New Season Mode",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🚀 New Season Mode: Composite Player Evaluation")

# =============================================================================
# 1. Default Profile & Session State
# =============================================================================
DEFAULT_PROFILE = {
    "weights": {
        "base_multiplier": 1.0,
        "talent_weight":   1.0,
        "buzz_weight":     1.0,
        "expert_weight":   1.0,
    },
    "mrb_params": {
        "GK": {"max_proportional_bonus_at_pvs100": 0.3},
        "DEF": {"max_proportional_bonus_at_pvs100": 0.4},
        "MID": {"max_proportional_bonus_at_pvs100": 0.6},
        "FWD": {"max_proportional_bonus_at_pvs100": 0.8},
    },
    "formation":        "4-4-2",
    "squad_size":       20,
    "min_recent_filter": 1,
}

# Initialize session state
for key, val in DEFAULT_PROFILE["weights"].items():
    if key not in st.session_state:
        st.session_state[key] = val

if "mrb_params" not in st.session_state:
    st.session_state.mrb_params = DEFAULT_PROFILE["mrb_params"].copy()

if "formation_key" not in st.session_state:
    st.session_state.formation_key = DEFAULT_PROFILE["formation"]
if "squad_size" not in st.session_state:
    st.session_state.squad_size = DEFAULT_PROFILE["squad_size"]
if "min_recent_filter" not in st.session_state:
    st.session_state.min_recent_filter = DEFAULT_PROFILE["min_recent_filter"]

# =============================================================================
# 2. Load Data (Cached)
# =============================================================================
@st.cache_data(show_spinner=False)
def load_data(uploaded) -> pd.DataFrame:
    if uploaded.name.lower().endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)

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

# =============================================================================
# 3. Auto‐Fill Missing Parameters
# =============================================================================
def fill_missing_params(df: pd.DataFrame) -> pd.DataFrame:
    df2 = df.copy()
    n = len(df2)
    if "talent_potential" not in df2:
        df2["talent_potential"] = np.random.uniform(4, 8, n).round(1)
    if "market_buzz" not in df2:
        df2["market_buzz"] = np.random.uniform(3, 9, n).round(1)
    if "expert_sentiment" not in df2:
        df2["expert_sentiment"] = np.random.uniform(5, 9, n).round(1)
    return df2

df_filled = fill_missing_params(df_raw)

# =============================================================================
# 4. Normalize Attributes (Cached)
# =============================================================================
@st.cache_data(show_spinner=False)
def normalize_player_attributes(df: pd.DataFrame) -> pd.DataFrame:
    dfn = df.copy()
    # Cote → 1–100
    dfn["Cote"] = pd.to_numeric(dfn["Cote"], errors="coerce").fillna(1).clip(1, 100)
    dfn["norm_cote"]   = dfn["Cote"]
    # 0–10 → 0–100
    dfn["norm_talent"] = dfn["talent_potential"].astype(float).clip(0,10) * 10
    dfn["norm_buzz"]   = dfn["market_buzz"].astype(float).clip(0,10)    * 10
    dfn["norm_expert"] = dfn["expert_sentiment"].astype(float).clip(0,10)* 10
    return dfn

df_norm = normalize_player_attributes(df_filled)

# =============================================================================
# 5. Sidebar: Composite Score & MRB Settings + Reset
# =============================================================================
with st.sidebar.expander("🎯 Composite Score Settings", expanded=True):
    st.slider(
        "Cote Weight",
        0.1, 5.0,
        key="base_multiplier",
        step=0.1,
        help="Weight for the player's current rating (0–100)."
    )
    st.slider(
        "Talent Potential Weight",
        0.1, 5.0,
        key="talent_weight",
        step=0.1,
        help="Weight for projected talent (0–10 scale)."
    )
    st.slider(
        "Market Buzz Weight",
        0.1, 5.0,
        key="buzz_weight",
        step=0.1,
        help="Weight for market hype (0–10 scale)."
    )
    st.slider(
        "Expert Sentiment Weight",
        0.1, 5.0,
        key="expert_weight",
        step=0.1,
        help="Weight for expert opinions (0–10 scale)."
    )

with st.sidebar.expander("💰 MRB Parameters"):
    for pos in ["GK", "DEF", "MID", "FWD"]:
        st.session_state.mrb_params[pos]["max_proportional_bonus_at_pvs100"] = st.slider(
            f"{pos} Max MRB Bonus @ PVS=100", 0.0, 1.0,
            value=st.session_state.mrb_params[pos]["max_proportional_bonus_at_pvs100"],
            step=0.01,
            help="Proportional bonus on top of Cote if PVS=100 (capped at 2×Cote)."
        )

with st.sidebar.expander("🔄 Controls"):
    if st.button("Reset to Default"):
        # Reset composite weights
        for k, v in DEFAULT_PROFILE["weights"].items():
            st.session_state[k] = v
        # Reset MRB params
        st.session_state.mrb_params = DEFAULT_PROFILE["mrb_params"].copy()
        # Reset squad settings
        st.session_state.formation_key    = DEFAULT_PROFILE["formation"]
        st.session_state.squad_size       = DEFAULT_PROFILE["squad_size"]
        st.session_state.min_recent_filter= DEFAULT_PROFILE["min_recent_filter"]
        st.experimental_rerun()

# =============================================================================
# 6. Calculate PVS (Cached)
# =============================================================================
@st.cache_data(show_spinner=False)
def calculate_pvs(
    df: pd.DataFrame,
    w_base: float,
    w_talent: float,
    w_buzz: float,
    w_expert: float
) -> pd.DataFrame:
    d = df.copy()
    total_w = w_base + w_talent + w_buzz + w_expert
    d["pvs"] = (
        d["norm_cote"] * w_base +
        d["norm_talent"] * w_talent +
        d["norm_buzz"]   * w_buzz +
        d["norm_expert"] * w_expert
    ) / max(total_w, 1e-6)
    return d.clip(lower=0, upper=100, axis=1)

df_pvs = calculate_pvs(
    df_norm,
    st.session_state.base_multiplier,
    st.session_state.talent_weight,
    st.session_state.buzz_weight,
    st.session_state.expert_weight
)

# =============================================================================
# 7. Calculate MRB & Value-per-Cost (Cached)
# =============================================================================
@st.cache_data(show_spinner=False)
def calculate_mrb(
    df: pd.DataFrame,
    mrb_params: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    d = df.copy()
    d["mrb"] = d["Cote"].astype(int)
    for pos, params in mrb_params.items():
        mask = d["simplified_position"] == pos
        if not mask.any():
            continue
        max_bonus = params["max_proportional_bonus_at_pvs100"]
        def _calc(row):
            base = row["Cote"]
            bonus = (row["pvs"] / 100) * max_bonus
            val = base * (1 + bonus)
            val = min(val, base * 2)
            return int(round(max(base, val)))
        d.loc[mask, "mrb"] = d.loc[mask].apply(_calc, axis=1)
    d["value_per_cost"] = d["pvs"] / d["mrb"].replace(0, np.nan)
    d["value_per_cost"].fillna(0, inplace=True)
    return d

df_evaluated = calculate_mrb(df_pvs, st.session_state.mrb_params)

# =============================================================================
# 8. Searchable Evaluated Table
# =============================================================================
st.subheader("2️⃣ Evaluated Players")
search = st.text_input("🔍 Search Evaluated Players")
df_display = df_evaluated.copy()
if search:
    df_display = df_display[
        df_display.apply(lambda r: r.astype(str).str.contains(search, case=False).any(), axis=1)
    ]
st.dataframe(df_display.reset_index(drop=True), use_container_width=True, height=300)
st.download_button(
    "📥 Download Evaluated Data",
    data=df_evaluated.to_csv(index=False).encode("utf-8"),
    file_name="evaluated_players.csv",
    mime="text/csv",
)

# =============================================================================
# 9. Squad Building (reuse your historical logic)
# =============================================================================
st.sidebar.markdown("---")
st.sidebar.markdown("🛡️ Squad Building Settings")

# Formation selector
strategist = MPGAuctionStrategist()
formation_key = st.sidebar.selectbox(
    "Formation",
    options=list(strategist.formations.keys()),
    index=list(strategist.formations.keys()).index(st.session_state.formation_key),
    key="formation_key",
)
# Squad size
squad_size = st.sidebar.number_input(
    "Squad Size",
    min_value=strategist.squad_minimums_sum_val,
    max_value=30,
    value=st.session_state.squad_size,
    key="squad_size",
)
# Min recent games filter
min_recent = st.sidebar.number_input(
    "Min Recent Games Played",
    min_value=0,
    max_value=st.session_state.min_recent_filter * 2 + 5,
    value=st.session_state.min_recent_filter,
    key="min_recent_filter",
)

# Build & display squad
squad_df, squad_summary = strategist.select_squad(
    df_evaluated,
    formation_key,
    squad_size,
    min_recent,
)

st.subheader("🏆 Suggested Squad")
st.dataframe(squad_df.reset_index(drop=True), use_container_width=True, height=300)

st.subheader("📈 Squad Summary")
st.write(squad_summary)
st.download_button(
    "📥 Download Squad CSV",
    data=squad_df.to_csv(index=False).encode("utf-8"),
    file_name="suggested_squad.csv",
    mime="text/csv",
)
