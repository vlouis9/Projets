# app.py

import streamlit as st

# Set page config immediately—this must be the first Streamlit command.
st.set_page_config(
    page_title="MPG Auction Strategist v4 (Optimized)",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Now import your mode modules.
# (Ensure that these modules do NOT call st.set_page_config themselves.)
from season_with_data import run_historical_mode
from new_season_evaluator import run_new_season_mode

# Selector UI: Choose either Historical Mode or New Season Mode.
mode = st.sidebar.selectbox(
    "Select Season Mode",
    ("Historical Season Data Mode", "New Season Mode (No Historical Data)")
)

if mode == "Historical Season Data Mode":
    run_historical_mode()
else:
    run_new_season_mode()
