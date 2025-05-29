# app.py
import streamlit as st
import pandas as pd

# Import functions for new season mode
from new_season_evaluator import (
    get_new_season_composite_evaluated_players_df,
    estimate_new_season_parameters
)

# Import your existing evaluation pipeline from your historical season module.
from season_with_data import MPGAuctionStrategist, load_and_preprocess_data

def main():
    st.title("MPG Auction Strategist - Hybrid Mode")
    
    # Select which season mode you want to run.
    mode = st.sidebar.selectbox(
        "Select Season Mode", 
        ("Historical Season Data Mode", "New Season Mode (No Historical Data)")
    )
    
    new_season_mode = (mode == "New Season Mode (No Historical Data)")
    st.sidebar.markdown(f"### {mode} Settings")
    
    # File uploader remains the same in both modes.
    uploaded_file = st.sidebar.file_uploader(
        "Upload MPG Ratings File (CSV/Excel)", 
        type=["csv", "xlsx", "xls"]
    )
    
    if not uploaded_file:
        st.info("Please upload your MPG ratings file.")
        return
    
    # Create an instance of your strategist to handle historical-data evaluations.
    strategist = MPGAuctionStrategist()
    
    # Load and preprocess data; in this function, the class instance is used to simplify positions, etc.
    df_processed = load_and_preprocess_data(uploaded_file, strategist)
    if df_processed is None or df_processed.empty:
        st.error("Data could not be processed or is empty!")
        return
    
    # --- Evaluate Players Based on the Chosen Mode ---
    if new_season_mode:
        # New Season Mode: use the composite evaluation using extra parameters.
        base_multiplier = st.sidebar.slider("Cote Weight", 0.1, 5.0, 1.0, 0.1)
        talent_weight  = st.sidebar.slider("Talent Potential Weight", 0.1, 5.0, 1.0, 0.1)
        buzz_weight    = st.sidebar.slider("Market Buzz Weight", 0.1, 5.0, 1.0, 0.1)
        expert_weight  = st.sidebar.slider("Expert Sentiment Weight", 0.1, 5.0, 1.0, 0.1)
        
        # Optionally, estimate the extra parameters if not present.
        df_processed = estimate_new_season_parameters(df_processed)
        
        df_evaluated = get_new_season_composite_evaluated_players_df(
            df_processed, 
            base_multiplier, talent_weight, buzz_weight, expert_weight
        )
    else:
        # Historical Season Data Mode: use your original evaluation pipeline.
        # For example, get settings from the sidebar.
        n_recent = st.sidebar.number_input("Recent Games Window", min_value=1, max_value=38, value=5)
        min_recent_filter = st.sidebar.number_input("Min Recent Filter", min_value=0, value=1)
        
        # You might also have profile settings for KPI weights and mrb params; here we use session_state defaults.
        # For this example, we use dummy parameters. In your code, these will be your more detailed settings.
        dummy_kpi_weights = {
            'GK': {'recent_avg': 0.05, 'season_avg': 0.70, 'calc_regularity': 0.25, 'recent_goals': 0.0, 'season_goals': 0.0},
            'DEF': {'recent_avg': 0.25, 'season_avg': 0.25, 'calc_regularity': 0.25, 'recent_goals': 0.0, 'season_goals': 0.0},
            'MID': {'recent_avg': 0.20, 'season_avg': 0.20, 'calc_regularity': 0.15, 'recent_goals': 0.15, 'season_goals': 0.15},
            'FWD': {'recent_avg': 0.15, 'season_avg': 0.15, 'calc_regularity': 0.10, 'recent_goals': 0.25, 'season_goals': 0.25}
        }
        dummy_mrb_params = {
            'GK': {'max_proportional_bonus_at_pvs100': 0.3},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.4},
            'MID': {'max_proportional_bonus_at_pvs100': 0.6}, 
            'FWD': {'max_proportional_bonus_at_pvs100': 0.8}
        }
        
        df_evaluated = strategist.get_evaluated_players_df(
            strategist,
            df_processed,
            n_recent,
            dummy_kpi_weights,
            dummy_mrb_params
        )
    
    st.subheader("Evaluated Players")
    st.dataframe(df_evaluated.head())
    
    # --- Squad Selection (Common to Both Modes) ---
    if not df_evaluated.empty:
        st.sidebar.markdown("### Squad Selection Settings")
        formation_key = st.sidebar.selectbox(
            "Select Formation", 
            options=list(strategist.formations.keys())
        )
        squad_size = st.sidebar.number_input(
            "Target Squad Size", 
            min_value=strategist.squad_minimums_sum_val, 
            max_value=30, 
            value=20
        )
        # Here we re-use the same min_recent_filter as in historical mode. For new season,
        # you might choose to bypass it or set another default.
        squad_df, summary = strategist.select_squad(df_evaluated, formation_key, squad_size, min_recent_filter)
        
        st.subheader("Suggested Squad")
        st.dataframe(squad_df)
        st.write("Squad Summary:", summary)

if __name__ == "__main__":
    main()
