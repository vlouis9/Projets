import streamlit as st

# Let the user select the mode.
mode = st.sidebar.selectbox(
    "Select Season Mode",
    ("Historical Season Data Mode", "New Season Mode (No Historical Data)")
)

if mode == "Historical Season Data Mode":
    from season_with_data import run_historical_mode
    run_historical_mode()
else:
    from new_season_evaluator import run_new_season_mode
    run_new_season_mode()
