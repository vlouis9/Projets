# new_season_evaluator.py
"""
Module: new_season_evaluator
Description: Provides functions to evaluate new season players using a composite
             performance metric (pvs) that blends a baseline 'Cote' with additional 
             factors such as talent potential, market buzz, and expert sentiment.
"""

import streamlit as st
import pandas as pd
import numpy as np

@st.cache_data
def get_new_season_composite_evaluated_players_df(
    df_processed: pd.DataFrame,
    base_multiplier: float,
    talent_weight: float,
    buzz_weight: float,
    expert_weight: float
) -> pd.DataFrame:
    """
    Evaluates new season players by calculating a composite new_PVS score.

    This function computes a composite performance rating ("pvs") for each player,
    by blending:
      - The normalized baseline rating ("Cote") using the base_multiplier.
      - A "talent potential" metric (expected future ability), 
      - A "market buzz" metric (hype/trend around the player), and 
      - An "expert sentiment" metric (scout/pundit evaluation).

    Parameters:
        df_processed : pd.DataFrame
            Pre-processed DataFrame containing at least a 'Cote' column.
            Optionally, it may include:
              - 'talent_potential' (expected on a 0–10 scale),
              - 'market_buzz' (0–10 scale),
              - 'expert_sentiment' (0–10 scale).
        base_multiplier : float
            Weight for the baseline rating (Cote).
        talent_weight : float
            Weight for the talent potential.
        buzz_weight : float
            Weight for the market buzz.
        expert_weight : float
            Weight for the expert sentiment.
            
    Returns:
        pd.DataFrame
            A DataFrame enriched with:
              - 'norm_cote': normalized cote (0–100)
              - 'norm_talent': talent potential normalized (0–100)
              - 'norm_buzz'  : market buzz normalized (0–100)
              - 'norm_expert': expert sentiment normalized (0–100)
              - 'pvs'        : composite performance rating (0–100)
              - 'mrb'        : market-based bid (here kept same as cote)
              - 'value_per_cost': ratio of pvs to mrb, used for squad selections.
    """

    if df_processed is None or df_processed.empty:
        return pd.DataFrame()

    # Create a working copy
    df_new = df_processed.copy()

    # Clean and convert the 'Cote' column: ensure numeric, fill missing with 1, and round.
    df_new['Cote'] = pd.to_numeric(df_new['Cote'], errors='coerce').fillna(1)
    df_new['Cote'] = df_new['Cote'].clip(lower=1).round().astype(int)
    # Assume cote is on a 0–100 scale; otherwise adapt the normalization appropriately.
    df_new['norm_cote'] = np.clip(df_new['Cote'], 0, 100)

    # Ensure the additional evaluation parameters exist.
    for col in ['talent_potential', 'market_buzz', 'expert_sentiment']:
        if col not in df_new.columns:
            df_new[col] = 5.0  # Default value in the middle of a 0–10 scale

    # Normalize the extra parameters (assumed to be on a 0–10 scale) into a 0–100 range.
    df_new['norm_talent'] = df_new['talent_potential'].astype(float) * 10.0
    df_new['norm_buzz']   = df_new['market_buzz'].astype(float) * 10.0
    df_new['norm_expert'] = df_new['expert_sentiment'].astype(float) * 10.0

    # Calculate the composite performance rating (pvs) as a weighted average.
    total_weight = base_multiplier + talent_weight + buzz_weight + expert_weight
    df_new['pvs'] = (
        df_new['norm_cote'] * base_multiplier +
        df_new['norm_talent'] * talent_weight +
        df_new['norm_buzz']   * buzz_weight +
        df_new['norm_expert'] * expert_weight
    ) / total_weight

    # Clip the final value to a maximum of 100.
    df_new['pvs'] = df_new['pvs'].clip(0, 100)

    # Define a baseline MRB (suggested bid) - here simply the cote.
    # You might further adjust this logic (for instance, applying a positional bonus).
    df_new['mrb'] = df_new['Cote']

    # Calculate a value per cost ratio to use in squad selection logic.
    # Replace 0 with NaN temporarily to avoid division by zero.
    df_new['value_per_cost'] = df_new['pvs'] / df_new['mrb'].replace(0, np.nan)
    df_new['value_per_cost'].fillna(0, inplace=True)

    return df_new

def estimate_new_season_parameters(players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Estimates additional parameters for new season players.

    This dummy implementation assigns random values to the parameters:
      - 'talent_potential': player's projected future ability (0–10)
      - 'market_buzz': a measure of hype/trend (0–10)
      - 'expert_sentiment': evaluation from scouts/experts (0–10)

    These values are placeholders and ideally should be replaced
    with data derived from scouting reports or external analytics.

    Parameters:
        players_df : pd.DataFrame
            DataFrame containing a list of players, at minimum with a "Cote" column.
    
    Returns:
        pd.DataFrame
            The input DataFrame augmented with the three parameters.
    """
    np.random.seed(42)  # For reproducibility during testing/demo
    n = len(players_df)
    # Talent potential estimated between 5 and 10.
    players_df['talent_potential'] = np.random.uniform(5, 10, n).round(1)
    # Market buzz estimated between 3 and 10.
    players_df['market_buzz'] = np.random.uniform(3, 10, n).round(1)
    # Expert sentiment estimated between 5 and 10.
    players_df['expert_sentiment'] = np.random.uniform(5, 10, n).round(1)
    return players_df
