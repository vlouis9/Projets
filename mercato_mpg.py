import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Set

# Page configuration
st.set_page_config(
    page_title="MPG Auction Strategist v5",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #004080; text-align: center; margin-bottom: 2rem; font-family: 'Roboto', sans-serif;}
    .section-header {font-size: 1.4rem; font-weight: bold; color: #006847; margin-top: 1.5rem; margin-bottom: 1rem; border-bottom: 2px solid #006847; padding-bottom: 0.3rem;}
    .stButton>button {background-color: #004080; color: white; font-weight: bold; border-radius: 0.3rem; padding: 0.4rem 0.8rem; border: none; width: 100%;}
    .stButton>button:hover {background-color: #003060; color: white;}
    .stSlider [data-baseweb="slider"] {padding-bottom: 12px;}
    .css-1d391kg {background-color: #f8f9fa; padding-top: 1rem;}
    .stExpander {border: 1px solid #e0e0e0; border-radius: 0.3rem; margin-bottom: 0.5rem;}
    /* Added some improvements for consistent look */
    h6 {font-size: 1.0rem; color: #333; margin-top: 0.5rem; margin-bottom: 0.2rem;}
</style>
""", unsafe_allow_html=True)

# Constants and Predefined Profiles
DEFAULT_N_RECENT_GAMES = 5
DEFAULT_MIN_RECENT_GAMES_PLAYED = 1
DEFAULT_SQUAD_SIZE = 20
DEFAULT_FORMATION = "4-4-2"
DEFAULT_BUDGET = 500 # Added default budget

PREDEFINED_PROFILES = {
    "Custom": "custom", # Marker for custom settings
    "Balanced Value": {
        "n_recent_games": 5,
        "min_recent_games_played_filter": 1,
        "kpi_weights": {
            'GK': {'recent_avg': 0.05, 'season_avg': 0.70, 'calc_regularity': 0.25, 'recent_goals': 0.0, 'season_goals': 0.0},
            'DEF': {'recent_avg': 0.25, 'season_avg': 0.25, 'calc_regularity': 0.25, 'recent_goals': 0.0, 'season_goals': 0.0},
            'MID': {'recent_avg': 0.20, 'season_avg': 0.20, 'calc_regularity': 0.15, 'recent_goals': 0.15, 'season_goals': 0.15},
            'FWD': {'recent_avg': 0.15, 'season_avg': 0.15, 'calc_regularity': 0.10, 'recent_goals': 0.25, 'season_goals': 0.25}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.3},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.4},
            'MID': {'max_proportional_bonus_at_pvs100': 0.6},
            'FWD': {'max_proportional_bonus_at_pvs100': 0.8}
        }
    },
    "Aggressive Bids (Pay for PVS)": {
        "n_recent_games": 5,
        "min_recent_games_played_filter": 0,
        "kpi_weights": {
            'GK': {'recent_avg': 0.35, 'season_avg': 0.35, 'calc_regularity': 0.20, 'recent_goals': 0.0, 'season_goals': 0.0},
            'DEF': {'recent_avg': 0.30, 'season_avg': 0.30, 'calc_regularity': 0.30, 'recent_goals': 0.0, 'season_goals': 0.0},
            'MID': {'recent_avg': 0.25, 'season_avg': 0.25, 'calc_regularity': 0.15, 'recent_goals': 0.15, 'season_goals': 0.15},
            'FWD': {'recent_avg': 0.25, 'season_avg': 0.25, 'calc_regularity': 0.10, 'recent_goals': 0.20, 'season_goals': 0.20}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 1.1},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.9},
            'MID': {'max_proportional_bonus_at_pvs100': 1.1},
            'FWD': {'max_proportional_bonus_at_pvs100': 1.5}
        }
    },
    "Focus on Recent Form": {
        "n_recent_games": 3,
        "min_recent_games_played_filter": 1,
        "kpi_weights": {
            'GK': {'recent_avg': 0.5, 'season_avg': 0.1, 'calc_regularity': 0.3, 'recent_goals': 0.0, 'season_goals': 0.0},
            'DEF': {'recent_avg': 0.4, 'season_avg': 0.1, 'calc_regularity': 0.4, 'recent_goals': 0.0, 'season_goals': 0.0},
            'MID': {'recent_avg': 0.4, 'season_avg': 0.1, 'calc_regularity': 0.15, 'recent_goals': 0.2, 'season_goals': 0.1},
            'FWD': {'recent_avg': 0.4, 'season_avg': 0.1, 'calc_regularity': 0.1, 'recent_goals': 0.25, 'season_goals': 0.1}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.6},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.5},
            'MID': {'max_proportional_bonus_at_pvs100': 0.6},
            'FWD': {'max_proportional_bonus_at_pvs100': 0.9}
        }
    },
    "Focus on Season Consistency": {
        "n_recent_games": 7,
        "min_recent_games_played_filter": 2,
        "kpi_weights": {
            'GK': {'recent_avg': 0.0, 'season_avg': 0.75, 'calc_regularity': 0.25, 'recent_goals': 0.0, 'season_goals': 0.0},
            'DEF': {'recent_avg': 0.0, 'season_avg': 0.75, 'calc_regularity': 0.15, 'recent_goals': 0.0, 'season_goals': 0.10},
            'MID': {'recent_avg': 0.0, 'season_avg': 0.6, 'calc_regularity': 0.1, 'recent_goals': 0.0, 'season_goals': 0.3},
            'FWD': {'recent_avg': 0.0, 'season_avg': 0.5, 'calc_regularity': 0.1, 'recent_goals': 0.0, 'season_goals': 0.4}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.9},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.8},
            'MID': {'max_proportional_bonus_at_pvs100': 0.9},
            'FWD': {'max_proportional_bonus_at_pvs100': 1.2}
        }
    }
}


class MPGAuctionStrategist:
    """
    A class to calculate player valuations (PVS, MRB) and suggest an optimal squad
    for MPG (Mon Petit Gazon) auctions based on various configurable parameters.
    """
    def __init__(self):
        self.formations = {
            "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
            "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
            "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
            "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
            "4-5-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
            "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},
            "5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1}
        }
        # MPG specific squad minimums
        self.squad_minimums = {"GK": 2, "DEF": 6, "MID": 6, "FWD": 4}
        self.budget = DEFAULT_BUDGET # Default budget, can be overridden if needed

    @property
    def squad_minimums_sum_val(self) -> int:
        """Returns the sum of minimum players required for each position."""
        return sum(self.squad_minimums.values())

    def simplify_position(self, position: str) -> str:
        """Simplifies raw position strings to standard categories (GK, DEF, MID, FWD)."""
        if pd.isna(position) or not isinstance(position, str) or position.strip() == '':
            return 'UNKNOWN'
        pos = position.upper().strip()
        if pos == 'G':
            return 'GK'
        elif pos in ['D', 'DL', 'DC']:
            return 'DEF'
        elif pos in ['M', 'MD', 'MO']:
            return 'MID'
        elif pos == 'A':
            return 'FWD'
        else:
            return 'UNKNOWN'

    def create_player_id(self, row: pd.Series) -> str:
        """Creates a unique player ID from player name, simplified position, and club."""
        name = str(row.get('Joueur', '')).strip()
        simplified_pos = row.get('simplified_position', 'UNKNOWN') # Use already simplified position
        club = str(row.get('Club', '')).strip()
        return f"{name}_{simplified_pos}_{club}"

    @st.cache_data
    def extract_rating_goals_starter(self, rating_str: str) -> Tuple[Optional[float], int, bool, bool]:
        """
        Extracts rating, goals, played status, and starter status from a gameweek rating string.
        Returns: (rating, goals, played_this_gw, is_starter)
        """
        if pd.isna(rating_str) or not isinstance(rating_str, str) or rating_str.strip() == '0' or rating_str.strip() == '':
            return None, 0, False, False

        val_str = rating_str.strip()
        goals = val_str.count('*')
        is_starter = '(' not in val_str
        clean_rating_str = re.sub(r'[()\*]', '', val_str)

        try:
            rating = float(clean_rating_str)
            return rating, goals, True, is_starter
        except ValueError:
            return None, 0, False, False

    @st.cache_data
    def get_gameweek_columns(self, df_columns: List[str]) -> List[str]:
        """Identifies and sorts gameweek columns (e.g., 'D1', 'D2')."""
        gw_cols_data = []
        for col in df_columns:
            match = re.fullmatch(r'D(\d+)', col)
            if match:
                gw_cols_data.append({'name': col, 'number': int(match.group(1))})
        return [col['name'] for col in sorted(gw_cols_data, key=lambda x: x['number'])]

    @st.cache_data(show_spinner=False)
    def calculate_kpis(self, df: pd.DataFrame, n_recent: int) -> pd.DataFrame:
        """
        Calculates key performance indicators (KPIs) for each player based on gameweek data.
        Optimized to use vectorized operations where possible.
        """
        rdf = df.copy()
        all_gws = self.get_gameweek_columns(df.columns)
        
        if not all_gws:
            st.warning("No gameweek columns (e.g., D1, D2) found in the uploaded file. KPI calculations will be based on 0s.")
            # Initialize with zeros if no gameweek data
            rdf['recent_avg_rating'] = 0.0
            rdf['season_avg_rating'] = 0.0
            rdf['recent_goals'] = 0
            rdf['season_goals'] = 0
            rdf['recent_games_played_count'] = 0
            rdf['calc_regularity_pct'] = 0.0
            rdf['games_started_season'] = 0
            rdf['total_season_gws_considered'] = 0
            return rdf

        # Pre-extract all ratings, goals, played, started status for all gameweeks
        # This creates columns like 'D1_rating', 'D1_goals', etc.
        for gw_col in all_gws:
            rdf[[f'{gw_col}_rating', f'{gw_col}_goals', f'{gw_col}_played', f'{gw_col}_starter']] = \
                rdf[gw_col].apply(lambda x: pd.Series(self.extract_rating_goals_starter(x)))
            
            # Ensure boolean columns are correctly typed
            rdf[f'{gw_col}_played'] = rdf[f'{gw_col}_played'].astype(bool)
            rdf[f'{gw_col}_starter'] = rdf[f'{gw_col}_starter'].astype(bool)
            # Ensure numerical columns are correctly typed
            rdf[f'{gw_col}_rating'] = pd.to_numeric(rdf[f'{gw_col}_rating'], errors='coerce').fillna(0.0)
            rdf[f'{gw_col}_goals'] = pd.to_numeric(rdf[f'{gw_col}_goals'], errors='coerce').fillna(0).astype(int)


        # Calculate season KPIs
        season_ratings_cols = [f'{gw}_rating' for gw in all_gws]
        season_goals_cols = [f'{gw}_goals' for gw in all_gws]
        season_started_cols = [f'{gw}_starter' for gw in all_gws]
        season_played_cols = [f'{gw}_played' for gw in all_gws]

        # Calculate season_avg_rating: average of non-zero ratings
        # Use a custom aggregation to handle NaNs correctly for mean (only average actual ratings)
        def safe_mean(series):
            valid_values = series[series > 0] # Only consider ratings > 0 for average
            return valid_values.mean() if not valid_values.empty else 0.0
        
        rdf['season_avg_rating'] = rdf[season_ratings_cols].apply(safe_mean, axis=1)
        rdf['season_goals'] = rdf[season_goals_cols].sum(axis=1)
        rdf['games_started_season'] = rdf[season_started_cols].sum(axis=1).astype(int)
        rdf['total_season_gws_considered'] = len(all_gws) # Total gameweeks in input file

        # Calculate regularity based on starts over total gameweeks
        rdf['calc_regularity_pct'] = (rdf['games_started_season'] / rdf['total_season_gws_considered'] * 100).fillna(0).clip(0, 100)
        
        # Calculate recent KPIs
        rec_gws_check = all_gws[-n_recent:]
        if not rec_gws_check: # Handle case where n_recent is larger than available gameweeks
            st.warning(f"Not enough gameweeks ({len(all_gws)}) for 'Recent Games Window' of {n_recent}. Recent KPIs will be 0.")
            rdf['recent_avg_rating'] = 0.0
            rdf['recent_goals'] = 0
            rdf['recent_games_played_count'] = 0
        else:
            recent_ratings_cols = [f'{gw}_rating' for gw in rec_gws_check]
            recent_goals_cols = [f'{gw}_goals' for gw in rec_gws_check]
            recent_played_cols = [f'{gw}_played' for gw in rec_gws_check]

            rdf['recent_avg_rating'] = rdf[recent_ratings_cols].apply(safe_mean, axis=1)
            rdf['recent_goals'] = rdf[recent_goals_cols].sum(axis=1)
            rdf['recent_games_played_count'] = rdf[recent_played_cols].sum(axis=1).astype(int)

        # Drop intermediate gameweek columns to clean up DataFrame
        cols_to_drop = [f'{gw_col}_{suffix}' for gw_col in all_gws for suffix in ['rating', 'goals', 'played', 'starter']]
        rdf.drop(columns=cols_to_drop, errors='ignore', inplace=True)

        return rdf

    @st.cache_data(show_spinner=False)
    def normalize_kpis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalizes KPI values to a 0-100 scale."""
        rdf = df.copy()
        rdf['norm_recent_avg'] = np.clip(rdf['recent_avg_rating'] * 10, 0, 100)
        rdf['norm_season_avg'] = np.clip(rdf['season_avg_rating'] * 10, 0, 100)
        
        # Note: '%Titu' column is kept for potential future use or display but not actively used in PVS calculation per provided logic.
        rdf['norm_regularity_file'] = pd.to_numeric(rdf['%Titu'], errors='coerce').fillna(0).clip(0, 100) 
        
        rdf['norm_calc_regularity'] = rdf['calc_regularity_pct'].clip(0, 100)

        rdf['norm_recent_goals'] = 0.0
        rdf['norm_season_goals'] = 0.0

        for pos in ['DEF', 'MID', 'FWD']: # Only these positions typically get goals
            mask = rdf['simplified_position'] == pos
            if mask.any():
                # Recent goals: simple multiplication. Max at 5 goals in 5 games = 100.
                rdf.loc[mask, 'norm_recent_goals'] = np.clip(rdf.loc[mask, 'recent_goals'] * 20, 0, 100)
                
                # Season goals: normalized by max goals in that position
                max_sg = rdf.loc[mask, 'season_goals'].max()
                if max_sg > 0:
                    rdf.loc[mask, 'norm_season_goals'] = np.clip(rdf.loc[mask, 'season_goals'] / max_sg * 100, 0, 100)
                else:
                    rdf.loc[mask, 'norm_season_goals'] = 0.0 # All zeros if no goals for that position

        return rdf

    @st.cache_data(show_spinner=False)
    def calculate_pvs(self, df: pd.DataFrame, weights: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """Calculates Player Value Score (PVS) based on normalized KPIs and positional weights."""
        rdf = df.copy()
        rdf['pvs'] = 0.0
        for pos, w in weights.items():
            mask = rdf['simplified_position'] == pos
            if not mask.any():
                continue

            # Calculate weighted sum for each KPI, handling potential missing weights with .get()
            pvs_sum = (
                rdf.loc[mask, 'norm_recent_avg'].fillna(0) * w.get('recent_avg', 0) +
                rdf.loc[mask, 'norm_season_avg'].fillna(0) * w.get('season_avg', 0) +
                rdf.loc[mask, 'norm_calc_regularity'].fillna(0) * w.get('calc_regularity', 0)
            )

            # Add goals only for relevant positions
            if pos in ['DEF', 'MID', 'FWD']:
                pvs_sum += (
                    rdf.loc[mask, 'norm_recent_goals'].fillna(0) * w.get('recent_goals', 0) +
                    rdf.loc[mask, 'norm_season_goals'].fillna(0) * w.get('season_goals', 0)
                )
            rdf.loc[mask, 'pvs'] = pvs_sum.clip(0, 100) # Ensure PVS is between 0 and 100
        return rdf

    @st.cache_data(show_spinner=False)
    def calculate_mrb(self, df: pd.DataFrame, mrb_params_per_pos: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        """
        Calculates Most Realistic Bid (MRB) based on player 'Cote' and PVS, with positional adjustments.
        MRB is capped at 2x Cote.
        """
        rdf = df.copy()
        # Ensure 'Cote' is numeric, fill NaNs with 1, clip to minimum 1, and convert to int
        rdf['Cote'] = pd.to_numeric(rdf['Cote'], errors='coerce').fillna(1).clip(lower=1).round().astype(int)
        rdf['mrb'] = rdf['Cote'] # Initialize MRB with Cote

        for pos_simplified, params in mrb_params_per_pos.items():
            mask = rdf['simplified_position'] == pos_simplified
            if not mask.any():
                continue
            
            max_prop_bonus = params.get('max_proportional_bonus_at_pvs100', 0.5)

            # Apply the MRB calculation for relevant players
            cotes = rdf.loc[mask, 'Cote']
            pvs_players = rdf.loc[mask, 'pvs']
            
            pvs_scaled_0_1 = pvs_players / 100.0
            pvs_derived_bonus_factor = pvs_scaled_0_1 * max_prop_bonus
            
            mrb_float = cotes * (1 + pvs_derived_bonus_factor)
            
            # Cap MRB at 2x Cote and ensure it's at least the Cote
            mrb_capped = np.minimum(mrb_float, cotes * 2)
            final_mrb_values = np.maximum(cotes, mrb_capped)
            
            rdf.loc[mask, 'mrb'] = final_mrb_values.round().astype(int)
        
        # Calculate value_per_cost
        rdf['value_per_cost'] = rdf['pvs'] / rdf['mrb'].replace(0, np.nan) # Replace 0 MRB with NaN for division
        rdf['value_per_cost'].fillna(0, inplace=True) # Fill NaNs (from 0 MRB) with 0

        return rdf

    def select_squad(self, df: pd.DataFrame, formation_key: str, target_squad_size: int,
                     min_recent_games_played: int, budget: int) -> Tuple[pd.DataFrame, Dict]:
        """
        Selects an optimal squad based on PVS, MRB, formation, and budget constraints.
        It prioritizes starters, then minimum positional requirements, then target squad size,
        and finally adjusts for budget.
        """
        if df.empty:
            return pd.DataFrame(), {}

        # --- Initial Filtering & Preparation ---
        # Filter out unavailable players and those not meeting recent games played criteria
        eligible_df = df[
            (~df['Indispo ?']) &
            (df['recent_games_played_count'] >= min_recent_games_played)
        ].copy()

        if eligible_df.empty:
            st.warning("No players are eligible after applying filters (unavailable, min recent games).")
            return pd.DataFrame(), {}
        
        # Ensure unique players based on player_id and convert MRB to int
        eligible_df = eligible_df.drop_duplicates(subset=['player_id']).copy()
        eligible_df['mrb'] = eligible_df['mrb'].astype(int)
        
        # Sort players by PVS (descending) for selection
        eligible_df = eligible_df.sort_values(by='pvs', ascending=False).reset_index(drop=True)

        # Initialize the current squad as a DataFrame
        # This DataFrame will hold players selected and their specific attributes like is_starter and mrb_actual_cost
        current_squad_df = pd.DataFrame(columns=[
            'player_id', 'mrb_actual_cost', 'pvs_in_squad', 'simplified_position', 'is_starter'
        ])

        formations_needed = self.formations[formation_key].copy()
        squad_min_needed = self.squad_minimums.copy()

        # Helper to check GK count without iterating a list of dicts
        def get_current_gk_count() -> int:
            return current_squad_df[current_squad_df['simplified_position'] == 'GK'].shape[0]
        
        # Helper to add a player to the current squad DataFrame
        def add_player_to_squad(player_row: pd.Series, is_starter_role: bool) -> bool:
            nonlocal current_squad_df
            if player_row['player_id'] in current_squad_df['player_id'].values:
                return False # Already in squad

            # GK constraint check
            if player_row['simplified_position'] == 'GK' and get_current_gk_count() >= 2:
                # st.caption(f"Cannot add GK {player_row['Joueur']}; already 2 GKs selected.")
                return False

            new_player_data = {
                'player_id': player_row['player_id'],
                'mrb_actual_cost': int(player_row['mrb']),
                'pvs_in_squad': float(player_row['pvs']),
                'simplified_position': player_row['simplified_position'],
                'is_starter': is_starter_role
            }
            # Using pd.concat for efficiency, though for single row appends, it's fine.
            current_squad_df = pd.concat([current_squad_df, pd.DataFrame([new_player_data])], ignore_index=True)
            return True

        # Helper to remove a player from the current squad DataFrame
        def remove_player_from_squad(player_id_to_remove: str) -> bool:
            nonlocal current_squad_df
            initial_len = current_squad_df.shape[0]
            current_squad_df = current_squad_df[current_squad_df['player_id'] != player_id_to_remove]
            return current_squad_df.shape[0] < initial_len
            
        # --- Phase A: Initial High-PVS Squad Construction (Potentially Over Budget) ---
        # A1: Select Starters
        for _, player_row in eligible_df.iterrows():
            pos = player_row['simplified_position']
            if len(current_squad_df) >= target_squad_size:
                break # Stop if target size is already met
            if formations_needed.get(pos, 0) > 0:
                if add_player_to_squad(player_row, True):
                    formations_needed[pos] -= 1
        
        # A2: Fulfill Overall Squad Positional Minimums
        for pos, min_needed in squad_min_needed.items():
            current_count_for_pos = current_squad_df[current_squad_df['simplified_position'] == pos].shape[0]
            while current_count_for_pos < min_needed and len(current_squad_df) < target_squad_size:
                candidate = eligible_df[
                    (eligible_df['simplified_position'] == pos) &
                    (~eligible_df['player_id'].isin(current_squad_df['player_id']))
                ].head(1)
                
                if candidate.empty:
                    break
                if add_player_to_squad(candidate.iloc[0], False):
                    current_count_for_pos += 1
                else: # Failed to add (e.g., 3rd GK)
                    # To prevent trying the same failing player, remove them from consideration.
                    eligible_df = eligible_df[eligible_df['player_id'] != candidate.iloc[0]['player_id']]
                    if eligible_df.empty: break # No more players to try
        
        # A3: Complete to Target Squad Size with best PVS players, respecting GK limit
        while len(current_squad_df) < target_squad_size:
            # Find best PVS player not already in squad, respecting GK limit
            candidate = eligible_df[
                ~eligible_df['player_id'].isin(current_squad_df['player_id'])
            ].head(1)

            if candidate.empty:
                break # No more eligible players
            
            if not add_player_to_squad(candidate.iloc[0], False):
                # If adding failed (likely 3rd GK), remove this player from eligible_df for future consideration
                eligible_df = eligible_df[eligible_df['player_id'] != candidate.iloc[0]['player_id']]
                if eligible_df.empty: break # No more players at all
                continue # Try to add next best player in next iteration

        current_total_mrb = current_squad_df['mrb_actual_cost'].sum()
        
        # --- Phase B: Iterative Budget Conformance (If Over Budget) ---
        max_budget_iterations_b = target_squad_size * 2 # Limit iterations to prevent infinite loops
        iterations_b_count = 0
        budget_conformance_tolerance = 1 # Aim to be within 1 unit of budget

        while current_total_mrb > budget + budget_conformance_tolerance and iterations_b_count < max_budget_iterations_b:
            iterations_b_count += 1
            made_a_downgrade_in_pass = False
            best_downgrade_action = None # (old_pid, new_pid, mrb_saved, pvs_change_val, new_player_row, original_starter_status)
            
            # Prioritize non-starters for downgrading, then starters
            squad_candidates_for_downgrade = current_squad_df.sort_values(
                by=['is_starter', 'mrb_actual_cost'], ascending=[True, False] # Non-starters first, then highest MRB
            )

            for _, old_player_dict_b in squad_candidates_for_downgrade.iterrows():
                old_pid_b = old_player_dict_b['player_id']
                old_pos_b = old_player_dict_b['simplified_position']
                old_mrb_b = old_player_dict_b['mrb_actual_cost']
                old_pvs_b = old_player_dict_b['pvs_in_squad']
                original_starter_status = old_player_dict_b['is_starter']

                # Find potential cheaper replacements for the same position, not already in squad
                potential_replacements_df = eligible_df[
                    (eligible_df['simplified_position'] == old_pos_b) &
                    (~eligible_df['player_id'].isin(current_squad_df['player_id'].drop(old_pid_b, errors='ignore'))) & # Exclude current squad except the one being replaced
                    (eligible_df['mrb'] < old_mrb_b)
                ].sort_values(by='pvs', ascending=False) # Get highest PVS among cheaper options

                if not potential_replacements_df.empty:
                    for _, new_player_row_b in potential_replacements_df.iterrows():
                        # Handle GK specific swap logic
                        if new_player_row_b['simplified_position'] == 'GK':
                            current_gk_count_b = get_current_gk_count()
                            # If old player was GK, and new player is also GK, count remains same.
                            # If old player was NOT GK, and new player IS GK, check if this adds a 3rd GK.
                            if old_pos_b != 'GK' and current_gk_count_b >= 2:
                                continue # Cannot add a 3rd GK
                            # If old player was GK, and current GK count is already 2 (meaning old player was one of them),
                            # this replacement is fine as count remains 2. No explicit check needed if old_pos_b == 'GK'.

                        mrb_saved_b = old_mrb_b - new_player_row_b['mrb']
                        pvs_change_val_b = new_player_row_b['pvs'] - old_pvs_b # new_pvs - old_pvs
                        
                        # Optimization metric: Maximize MRB saved, then minimize PVS loss (or maximize PVS gain)
                        # We want a positive mrb_saved and as positive pvs_change_val as possible.
                        # Using a weighted score: mrb_saved * weight_mrb + pvs_change_val * weight_pvs
                        # For now, prioritize MRB saved, then PVS change.
                        if best_downgrade_action is None or \
                           (mrb_saved_b > best_downgrade_action[2]) or \
                           (mrb_saved_b == best_downgrade_action[2] and pvs_change_val_b > best_downgrade_action[3]):
                            best_downgrade_action = (old_pid_b, new_player_row_b['player_id'], mrb_saved_b, pvs_change_val_b, new_player_row_b, original_starter_status)
            
            if best_downgrade_action:
                old_id_exec, new_id_exec, mrb_s_exec, pvs_c_exec, new_player_data_exec, original_starter_status_exec = best_downgrade_action
                
                # Perform the swap
                if remove_player_from_squad(old_id_exec):
                    if add_player_to_squad(new_player_data_exec, original_starter_status_exec):
                        current_total_mrb = current_squad_df['mrb_actual_cost'].sum()
                        st.caption(f"Budget Downgrade: Swapped '{eligible_df[eligible_df['player_id'] == old_id_exec]['Joueur'].iloc[0]}' for '{new_player_data_exec['Joueur']}'. Saved €{mrb_s_exec}. PVS change: {pvs_c_exec:.2f}. New MRB: {current_total_mrb}")
                        made_a_downgrade_in_pass = True
                    else:
                        st.warning(f"Failed to add replacement '{new_player_data_exec['Joueur']}' during downgrade (likely GK constraint). Re-adding original player.")
                        # If replacement fails, re-add original player to maintain squad integrity
                        old_player_data = eligible_df[eligible_df['player_id'] == old_id_exec].iloc[0]
                        add_player_to_squad(old_player_data, original_starter_status_exec)
                        break # Stop trying downgrades for this pass
                else:
                    st.warning(f"Failed to remove '{old_id_exec}' during downgrade. Skipping this swap.")
                    break # Critical error, stop trying to downgrade
            
            if not made_a_downgrade_in_pass:
                if current_total_mrb > budget + budget_conformance_tolerance:
                     st.warning(f"Budget Target Not Met: Current MRB {current_total_mrb} > Budget {budget}. No more effective downgrades found.")
                break # No beneficial downgrades found in this full pass

        # --- Phase C: Final PVS Upgrade (Spend Remaining Budget) ---
        budget_left_for_upgrades = budget - current_total_mrb
        max_upgrade_passes_c = target_squad_size
        upgrade_pass_count_c = 0
        
        while budget_left_for_upgrades > 5 and upgrade_pass_count_c < max_upgrade_passes_c and current_squad_df.shape[0] == target_squad_size:
            upgrade_pass_count_c += 1
            made_an_upgrade_this_pass_c = False
            best_upgrade_action_c = None # (old_pid, new_pid, mrb_increase, pvs_gain, new_player_row, original_starter_status)

            # Candidates for upgrade: prioritize non-starters with lower PVS, then starters with lower PVS
            squad_for_upgrade_cands_c = current_squad_df.sort_values(
                by=['is_starter', 'pvs_in_squad'], ascending=[True, True] # Non-starters first, then lowest PVS
            )

            for _, old_player_dict_c in squad_for_upgrade_cands_c.iterrows():
                old_pid_c = old_player_dict_c['player_id']
                old_pos_c = old_player_dict_c['simplified_position']
                old_mrb_c = old_player_dict_c['mrb_actual_cost']
                old_pvs_c = old_player_dict_c['pvs_in_squad']
                original_starter_status_c = old_player_dict_c['is_starter']

                # Find potential upgrades for the same position, not already in squad
                potential_upgrades_df = eligible_df[
                    (eligible_df['simplified_position'] == old_pos_c) &
                    (~eligible_df['player_id'].isin(current_squad_df['player_id'].drop(old_pid_c, errors='ignore'))) &
                    (eligible_df['pvs'] > old_pvs_c) # Must be PVS improvement
                ].sort_values(by='pvs', ascending=False) # Get highest PVS upgrades first

                for _, new_player_row_c in potential_upgrades_df.iterrows():
                    # Handle GK specific swap logic
                    if new_player_row_c['simplified_position'] == 'GK':
                        current_gk_count_c = get_current_gk_count()
                        if old_pos_c != 'GK' and current_gk_count_c >= 2:
                            continue # Cannot add a 3rd GK
                        # If old player was GK and new player is GK, no count change.

                    mrb_increase_c = new_player_row_c['mrb'] - old_mrb_c
                    pvs_gain_c = new_player_row_c['pvs'] - old_pvs_c

                    if mrb_increase_c <= budget_left_for_upgrades: # Must be affordable
                        # Metric: Maximize PVS gain per MRB increase (PVS/Cost efficiency)
                        current_upgrade_score = pvs_gain_c / (mrb_increase_c + 0.1) if mrb_increase_c > 0 else (pvs_gain_c * 100) # Heuristic for mrb_increase=0

                        if best_upgrade_action_c is None or \
                           current_upgrade_score > (best_upgrade_action_c[3] / (best_upgrade_action_c[2] + 0.1)):
                            best_upgrade_action_c = (old_pid_c, new_player_row_c['player_id'], mrb_increase_c, pvs_gain_c, new_player_row_c, original_starter_status_c)
            
            if best_upgrade_action_c:
                old_id_exec_c, new_id_exec_c, mrb_inc_exec_c, pvs_g_exec_c, new_player_data_exec_c, original_starter_status_c = best_upgrade_action_c
                
                # Perform the swap
                if remove_player_from_squad(old_id_exec_c):
                    if add_player_to_squad(new_player_data_exec_c, original_starter_status_c):
                        current_total_mrb = current_squad_df['mrb_actual_cost'].sum()
                        budget_left_for_upgrades = budget - current_total_mrb
                        st.caption(f"Budget Upgrade: Swapped '{eligible_df[eligible_df['player_id'] == old_id_exec_c]['Joueur'].iloc[0]}' for '{new_player_data_exec_c['Joueur']}'. MRB increase €{mrb_inc_exec_c}. PVS gain: {pvs_g_exec_c:.2f}. New Total MRB: {current_total_mrb}")
                        made_an_upgrade_this_pass_c = True
                    else:
                        st.warning(f"Failed to add replacement '{new_player_data_exec_c['Joueur']}' during upgrade. Re-adding original player.")
                        old_player_data = eligible_df[eligible_df['player_id'] == old_id_exec_c].iloc[0]
                        add_player_to_squad(old_player_data, original_starter_status_c)
                        break # Stop trying upgrades for this pass
                else:
                    st.warning(f"Failed to remove '{old_id_exec_c}' during upgrade. Skipping this swap.")
                    break # Critical error, stop trying to upgrade
            
            if not made_an_upgrade_this_pass_c:
                break # No more beneficial upgrades found

        # --- Final Squad Dataframe & Summary ---
        if current_squad_df.empty:
            return pd.DataFrame(), {}
        
        # Merge the selected squad data with the original processed DataFrame to get all player details
        final_squad_df = pd.merge(current_squad_df, eligible_df.drop(columns=['mrb', 'pvs']), 
                                  on=['player_id', 'simplified_position'], how='left')
        
        # Re-determine starter status based on PVS within formation constraints for the *final* squad
        final_starter_ids_definitive = set()
        temp_formation_needs_final = self.formations[formation_key].copy()
        
        # Sort the final squad players by PVS to pick the best for starter roles
        final_squad_df_sorted_for_final_starters = final_squad_df.sort_values(by='pvs_in_squad', ascending=False)

        for _, player_row_final_pass in final_squad_df_sorted_for_final_starters.iterrows():
            pos_final_pass = player_row_final_pass['simplified_position']
            player_id_final_pass = player_row_final_pass['player_id']
            if temp_formation_needs_final.get(pos_final_pass, 0) > 0:
                final_starter_ids_definitive.add(player_id_final_pass)
                temp_formation_needs_final[pos_final_pass] -= 1
        
        final_squad_df['is_starter'] = final_squad_df['player_id'].isin(final_starter_ids_definitive)

        final_total_mrb_actual = final_squad_df['mrb_actual_cost'].sum()
        summary = {
            'total_players': len(final_squad_df),
            'total_cost': int(final_total_mrb_actual),
            'remaining_budget': int(budget - final_total_mrb_actual),
            'position_counts': final_squad_df['simplified_position'].value_counts().to_dict(),
            'total_squad_pvs': round(final_squad_df['pvs_in_squad'].sum(), 2),
            'total_starters_pvs': round(final_squad_df[final_squad_df['is_starter']]['pvs_in_squad'].sum(), 2)
        }
        
        # Validation checks
        final_pos_counts_check_final = summary['position_counts']
        for pos_check, min_val_check in self.squad_minimums.items():
            if final_pos_counts_check_final.get(pos_check,0) < min_val_check:
                st.error(f"Squad Selection Issue: Position **{pos_check}** minimum not met! ({final_pos_counts_check_final.get(pos_check,0)}/{min_val_check})")
        if len(final_squad_df) != target_squad_size :
             st.error(f"Squad Selection Issue: Final squad size **{len(final_squad_df)}** does not match target **{target_squad_size}**.")

        return final_squad_df, summary


def main():
    st.markdown('<h1 class="main-header">🚀 MPG Auction Strategist v5 (Budget Focus)</h1>', unsafe_allow_html=True)
    strategist = MPGAuctionStrategist()

    # --- Initialize Streamlit Session State ---
    # This ensures consistency across reruns and allows widgets to control state.
    if 'current_profile_name' not in st.session_state:
        st.session_state.current_profile_name = "Balanced Value"
    if 'n_recent' not in st.session_state:
        st.session_state.n_recent = DEFAULT_N_RECENT_GAMES
    if 'min_recent_filter' not in st.session_state:
        st.session_state.min_recent_filter = DEFAULT_MIN_RECENT_GAMES_PLAYED
    if 'kpi_weights' not in st.session_state:
        st.session_state.kpi_weights = PREDEFINED_PROFILES["Balanced Value"]["kpi_weights"].copy()
    if 'mrb_params_per_pos' not in st.session_state:
        st.session_state.mrb_params_per_pos = PREDEFINED_PROFILES["Balanced Value"]["mrb_params_per_pos"].copy()
    if 'formation_key' not in st.session_state:
        st.session_state.formation_key = DEFAULT_FORMATION
    if 'squad_size' not in st.session_state:
        st.session_state.squad_size = DEFAULT_SQUAD_SIZE
    if 'budget' not in st.session_state:
        st.session_state.budget = DEFAULT_BUDGET
    
    # --- Sidebar UI Elements ---
    st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFr1EWtMR2tHq1FwHnCHqg2uXv1JMLYQlRZw&s", width=100)
    st.sidebar.markdown('<h2 class="section-header" style="margin-top:0;">⚙️ Controls & Settings</h2>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("📁 Upload MPG Ratings File (CSV/Excel)", type=['csv', 'xlsx', 'xls'], help="Joueur, Poste, Club, Cote, %Titu, Indispo?, Gameweeks (D1..D34).")

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🎨 Settings Profiles")
    profile_names = list(PREDEFINED_PROFILES.keys())

    def apply_profile_settings(profile_name):
        # Update current profile name state FIRST
        st.session_state.current_profile_name = profile_name
        if profile_name != "Custom" and profile_name in PREDEFINED_PROFILES:
            profile = PREDEFINED_PROFILES[profile_name]
            st.session_state.n_recent = profile.get("n_recent_games", st.session_state.n_recent) 
            st.session_state.min_recent_filter = profile.get("min_recent_games_played_filter", st.session_state.min_recent_filter)
            st.session_state.kpi_weights = profile.get("kpi_weights", {}).copy() # Use .copy() to prevent direct modification of profile dicts
            st.session_state.mrb_params_per_pos = profile.get("mrb_params_per_pos", {}).copy()
        # If "Custom", the existing session state values (potentially modified by user) are retained.

    selected_profile_name_ui = st.sidebar.selectbox("Select Profile", options=profile_names, 
                                                    index=profile_names.index(st.session_state.current_profile_name), 
                                                    key="profile_selector_v5_main_unique",
                                                    help="Loads predefined settings. Modifying details below sets to 'Custom'.")
    if selected_profile_name_ui != st.session_state.current_profile_name:
        apply_profile_settings(selected_profile_name_ui)
        st.rerun() # Trigger a rerun to apply profile settings immediately

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🌎 Global Data & Form Parameters")
    
    # Use explicit keys for all widgets to prevent issues with Streamlit re-rendering
    n_recent_ui = st.sidebar.number_input("Recent Games Window (N)", min_value=1, max_value=38, 
                                          value=st.session_state.n_recent, 
                                          key='n_recent_input_v5',
                                          help="For 'Recent Form' KPIs. Avg of games *played* in this window.")
    min_recent_filter_ui = st.sidebar.number_input("Filter: Min Games Played in Recent N Weeks", min_value=0, max_value=n_recent_ui, 
                                                   value=st.session_state.min_recent_filter, 
                                                   key='min_recent_filter_input_v5',
                                                   help=f"Exclude players with < this in '{n_recent_ui}' recent weeks. 0 = no filter.")
    
    # Check if these global params changed, if so, set profile to "Custom"
    if n_recent_ui != st.session_state.n_recent or min_recent_filter_ui != st.session_state.min_recent_filter:
        st.session_state.current_profile_name = "Custom"
    st.session_state.n_recent = n_recent_ui
    st.session_state.min_recent_filter = min_recent_filter_ui

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 👥 Squad Building Parameters")
    formation_key_ui = st.sidebar.selectbox("Preferred Starting Formation", options=list(strategist.formations.keys()), 
                                            index=list(strategist.formations.keys()).index(st.session_state.formation_key),
                                            key='formation_key_select_v5')
    target_squad_size_ui = st.sidebar.number_input("Target Total Squad Size", min_value=strategist.squad_minimums_sum_val, max_value=30, 
                                                 value=st.session_state.squad_size,
                                                 key='target_squad_size_input_v5')
    budget_ui = st.sidebar.number_input("Total Auction Budget", min_value=100, max_value=2000, 
                                        value=st.session_state.budget,
                                        key='budget_input_v5', help="Your total budget for the auction.")

    # Check if squad building params changed, if so, set profile to "Custom"
    if formation_key_ui != st.session_state.formation_key or \
       target_squad_size_ui != st.session_state.squad_size or \
       budget_ui != st.session_state.budget:
        st.session_state.current_profile_name = "Custom"
    st.session_state.formation_key = formation_key_ui
    st.session_state.squad_size = target_squad_size_ui
    st.session_state.budget = budget_ui
    strategist.budget = budget_ui # Update the strategist instance's budget

    with st.sidebar.expander("📊 KPI Weights (Click to Customize)", expanded=(st.session_state.current_profile_name == "Custom")):
        # Create a temporary dict for UI changes to compare with session state later
        weights_ui = {}
        for pos_key in ['GK', 'DEF', 'MID', 'FWD']:
            st.markdown(f'<h6>{pos_key}</h6>', unsafe_allow_html=True)
            current_pos_w_vals = st.session_state.kpi_weights.get(pos_key, {}) # Get current from session state

            weights_ui[pos_key] = {
                'season_avg': st.slider(f"{pos_key} Season Avg Rating", 0.0, 1.0, float(current_pos_w_vals.get('season_avg', 0.0)), 0.01, key=f"{pos_key}_wSA_v5_final"),
                'season_goals': st.slider(f"{pos_key} Season Goals", 0.0, 1.0, float(current_pos_w_vals.get('season_goals', 0.0)) if pos_key in ['DEF', 'MID', 'FWD'] else 0.0, 0.01, key=f"{pos_key}_wSG_v5_final", disabled=pos_key not in ['DEF','MID', 'FWD']),
                'calc_regularity': st.slider(f"{pos_key} Calculated Regularity", 0.0, 1.0, float(current_pos_w_vals.get('calc_regularity', 0.0)), 0.01, key=f"{pos_key}_wCR_v5_final", help="Based on starts identified in gameweek data."),
                'recent_goals': st.slider(f"{pos_key} Recent Goals", 0.0, 1.0, float(current_pos_w_vals.get('recent_goals', 0.0)) if pos_key in ['DEF', 'MID', 'FWD'] else 0.0, 0.01, key=f"{pos_key}_wRG_v5_final", disabled=pos_key not in ['DEF','MID', 'FWD']),
                'recent_avg': st.slider(f"{pos_key} Recent Avg Rating", 0.0, 1.0, float(current_pos_w_vals.get('recent_avg', 0.0)), 0.01, key=f"{pos_key}_wRA_v5_final"),
            }
        
        # Important: Check for changes only after all sliders have been rendered and their values collected.
        # This comparison will trigger 'Custom' profile if any slider is moved.
        if weights_ui != st.session_state.kpi_weights:
            st.session_state.current_profile_name = "Custom"
            st.session_state.kpi_weights = weights_ui.copy() # Update session state with the new values

    with st.sidebar.expander("💰 MRB Parameters (Click to Customize)", expanded=(st.session_state.current_profile_name == "Custom")):
        mrb_params_ui = {}
        for pos_key in ['GK', 'DEF', 'MID', 'FWD']:
            st.markdown(f'<h6>{pos_key}</h6>', unsafe_allow_html=True)
            current_pos_mrb_vals = st.session_state.mrb_params_per_pos.get(pos_key, {})

            mrb_params_ui[pos_key] = {
                'max_proportional_bonus_at_pvs100': st.slider(f"{pos_key} Max Bonus Factor (at PVS 100)", 0.0, 1.0, # Range corrected to match internal capping
                                                              float(current_pos_mrb_vals.get('max_proportional_bonus_at_pvs100', 0.2)), 
                                                              0.01, key=f"{pos_key}_mrbMPB_v5_final", 
                                                              help="Bonus factor if PVS=100 (e.g., 0.5 = 50% bonus implies MRB up to 1.5x Cote). Overall MRB is capped at 2x Cote.")
            }
        if mrb_params_ui != st.session_state.mrb_params_per_pos:
            st.session_state.current_profile_name = "Custom"
            st.session_state.mrb_params_per_pos = mrb_params_ui.copy() # Update session state


    # --- Dynamic Calculation and Display ---
    if uploaded_file:
        with st.spinner("🧠 Strategizing your optimal squad..."):
            try:
                # Load data
                df_input_calc = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded_file)
                
                # Initial processing: simplify position, create player ID, clean 'Cote', handle 'Indispo ?'
                df_processed_calc = df_input_calc.copy()
                df_processed_calc['simplified_position'] = df_processed_calc['Poste'].apply(strategist.simplify_position)
                df_processed_calc['player_id'] = df_processed_calc.apply(strategist.create_player_id, axis=1)
                df_processed_calc['Cote'] = pd.to_numeric(df_processed_calc['Cote'], errors='coerce').fillna(1).clip(lower=1).round().astype(int)
                
                if 'Indispo ?' not in df_processed_calc.columns:
                    df_processed_calc['Indispo ?'] = False
                else:
                    df_processed_calc['Indispo ?'] = df_processed_calc['Indispo ?'].astype(str).str.upper().isin(['TRUE', 'OUI', '1', 'YES', 'VRAI', 'INDISPO', 'O'])

                # Calculate KPIs, normalize, PVS, MRB
                df_kpis = strategist.calculate_kpis(df_processed_calc, st.session_state.n_recent)
                df_norm_kpis = strategist.normalize_kpis(df_kpis)
                df_pvs = strategist.calculate_pvs(df_norm_kpis, st.session_state.kpi_weights)
                df_mrb = strategist.calculate_mrb(df_pvs, st.session_state.mrb_params_per_pos)
                
                # Squad selection
                squad_df_result, squad_summary_result = strategist.select_squad(
                    df_mrb, st.session_state.formation_key, st.session_state.squad_size,
                    st.session_state.min_recent_filter, st.session_state.budget
                )
                
                # Store results in session state for display
                st.session_state['df_for_display_final'] = df_mrb
                st.session_state['squad_df_result_final'] = squad_df_result
                st.session_state['squad_summary_result_final'] = squad_summary_result
                st.session_state['selected_formation_key_display_final'] = st.session_state.formation_key

            except Exception as e:
                st.error(f"💥 An error occurred during data processing or squad selection: {str(e)}")
                st.exception(e) # Show traceback for debugging

        # --- Main Panel Display Logic ---
        if 'squad_df_result_final' in st.session_state and \
           st.session_state['squad_df_result_final'] is not None and \
           not st.session_state['squad_df_result_final'].empty:
            
            col_main_results, col_summary = st.columns([3, 1])
            with col_main_results:
                st.markdown('<h2 class="section-header">🏆 Suggested Squad</h2>', unsafe_allow_html=True)
                sdf = st.session_state['squad_df_result_final'].copy()
                
                # Ensure numerical columns are correctly typed for display
                int_cols_squad = ['mrb_actual_cost', 'Cote', 'recent_goals', 'season_goals', 'recent_games_played_count', 'games_started_season']
                for col in int_cols_squad:
                    if col in sdf.columns: 
                        sdf[col] = pd.to_numeric(sdf[col], errors='coerce').fillna(0).round().astype(int)
                
                # Select and rename columns for display in the suggested squad table
                squad_cols_display = [
                    'Joueur', 'Club', 'simplified_position', 'is_starter', 'pvs_in_squad',
                    'Cote', 'mrb_actual_cost', 'value_per_cost',
                    'season_avg_rating', 'season_goals', 'calc_regularity_pct',
                    'recent_avg_rating', 'recent_goals', 'recent_games_played_count'
                ]
                # Filter for columns that actually exist in the DataFrame
                squad_cols_exist_display = [col for col in squad_cols_display if col in sdf.columns]
                sdf = sdf[squad_cols_exist_display]
                
                sdf.rename(columns={
                    'Joueur': 'Player', 'simplified_position': 'Pos', 'pvs_in_squad': 'PVS',
                    'Cote': 'Cote', 'mrb_actual_cost': 'Suggested Bid', 'season_avg_rating': 'AvgR',
                    'season_goals': 'Goals', 'calc_regularity_pct': '%Reg',
                    'recent_goals': 'Rec.G', 'recent_avg_rating': 'Rec.AvgR', 
                    'value_per_cost': 'Val/Bid', 'is_starter': 'Starter'
                }, inplace=True)
                
                # Format float columns for display
                float_cols_squad_format = ['PVS', 'AvgR', '%Reg', 'Rec.AvgR', 'Val/Bid']
                for col in float_cols_squad_format: 
                    if col in sdf.columns:
                        sdf[col] = pd.to_numeric(sdf[col], errors='coerce').fillna(0.0).round(2)
                
                # Ensure positional order and sort the squad DataFrame
                pos_order = ['GK', 'DEF', 'MID', 'FWD']
                if 'Pos' in sdf.columns:
                    sdf['Pos'] = pd.Categorical(sdf['Pos'], categories=pos_order, ordered=True)
                    sdf = sdf.sort_values(by=['Starter', 'Pos', 'PVS'], ascending=[False, True, False])
                
                st.dataframe(sdf, use_container_width=True, hide_index=True)

            with col_summary:
                st.markdown('<h2 class="section-header">📈 Squad Summary</h2>', unsafe_allow_html=True)
                summary = st.session_state['squad_summary_result_final']
                if summary and isinstance(summary, dict):
                    st.metric("Budget Spent (MRB)", f"€ {summary.get('total_cost', 0):.0f}", help=f"Remaining: € {summary.get('remaining_budget', 0):.0f}")
                    st.metric("Squad Size", f"{summary.get('total_players', 0)} (Target: {st.session_state.squad_size})")
                    st.metric("Total Squad PVS", f"{summary.get('total_squad_pvs', 0):.2f}")
                    st.metric("Starters PVS", f"{summary.get('total_starters_pvs', 0):.2f}")
                    st.info(f"**Formation:** {st.session_state.get('selected_formation_key_display_final', 'N/A')}")
                    st.markdown("**Positional Breakdown:**")
                    for pos_cat_sum in pos_order:
                        count_sum = summary.get('position_counts', {}).get(pos_cat_sum, 0)
                        min_req_sum = strategist.squad_minimums.get(pos_cat_sum, 0)
                        st.markdown(f"• **{pos_cat_sum}:** {count_sum} (Min: {min_req_sum})")
                else:
                    st.warning("Squad summary unavailable.")
            
            st.markdown('<hr><h2 class="section-header">📋 Full Player Database & Values</h2>', unsafe_allow_html=True)
            if 'df_for_display_final' in st.session_state and st.session_state['df_for_display_final'] is not None:
                df_full = st.session_state['df_for_display_final'].copy()
                
                # Ensure numerical columns are correctly typed for the full display
                int_cols_full_display = ['Cote', 'mrb', 'recent_goals', 'season_goals', 'recent_games_played_count', 'games_started_season', 'total_season_gws_considered']
                for col in int_cols_full_display:
                    if col in df_full.columns:
                        df_full[col] = pd.to_numeric(df_full[col], errors='coerce').fillna(0).round().astype(int)
                
                # Select and rename columns for the full database table
                all_stats_cols_display = [
                    'Joueur', 'Club', 'simplified_position', 'pvs', 'Cote', 'mrb', 'value_per_cost', 'Indispo ?',
                    'season_avg_rating', 'season_goals', 'calc_regularity_pct', 'games_started_season',
                    'recent_avg_rating', 'recent_goals', 'recent_games_played_count'
                ]
                df_full = df_full[[col for col in all_stats_cols_display if col in df_full.columns]]
                
                df_full.rename(columns={
                    'Joueur': 'Player', 'simplified_position': 'Pos', 'pvs': 'PVS', 'Cote': 'Cote',
                    'mrb': 'Suggested Bid', 'value_per_cost': 'Val/Bid', 'Indispo ?': 'Unavail.',
                    'season_avg_rating': 'AvgR', 'season_goals': 'Goals', 'calc_regularity_pct': '%Reg',
                    'games_started_season': 'Sea.Start', 'recent_avg_rating': 'Rec.AvgR',
                    'recent_goals': 'Rec.G', 'recent_games_played_count': 'Rec.Plyd'
                }, inplace=True)
                
                # Format float columns for the full list
                float_cols_full_format = ['PVS', 'AvgR', '%Reg', 'Rec.AvgR', 'Val/Bid']
                for col in float_cols_full_format:
                    if col in df_full.columns:
                        df_full[col] = pd.to_numeric(df_full[col], errors='coerce').fillna(0.0).round(2)

                search_all = st.text_input("🔍 Search All Players:", key="search_all_v5")
                if search_all:
                    df_full = df_full[df_full.apply(lambda r: r.astype(str).str.contains(search_all, case=False, na=False).any(), axis=1)]
                
                st.dataframe(df_full.sort_values(by='PVS', ascending=False), use_container_width=True, hide_index=True, height=600)
                st.download_button(
                    label="📥 Download Full Analysis (CSV)",
                    data=df_full.to_csv(index=False).encode('utf-8'),
                    file_name="mpg_full_player_analysis_v5.csv",
                    mime="text/csv",
                    key="download_v5"
                )
            else:
                st.info("📊 Adjust settings in the sidebar. Results will appear here once loaded.")
        
        else: # This 'else' covers cases where processing failed or no squad was selected
             st.warning("No squad could be generated with the current settings and data. Please check filters, data quality, or try different parameters.")
    
    else: # This 'else' is for when no file has been uploaded yet
        st.info("👈 Upload your MPG ratings file to begin.")
        st.markdown('<hr><h2 class="section-header">📋 Expected File Format Guide</h2>', unsafe_allow_html=True)
        example_data = {
            'Joueur': ['Player A', 'Player B', 'Player C'], 
            'Poste': ['A', 'M', 'G'], 
            'Club': ['Club X', 'Club Y', 'Club Z'], 
            'Cote': [45, 30, 20], 
            '%Titu': [90, 75, 60], 
            'Indispo ?': ['', 'TRUE', ''], # Example unavailable player
            'D34': ['7.5*', '(6.0)**', '5.0'], # Rating, goals, sub/starter
            'D33': ['6.5', '0', '(4.5)'],
            'D32': ['5.0*', '7.0', '']
        }
        st.dataframe(pd.DataFrame(example_data), use_container_width=True, hide_index=True)
        st.markdown("""
        **Key Column Explanations:**
        - **Joueur, Poste, Club, Cote, %Titu**: Standard MPG player data.
        - **Indispo ?**: Mark players as unavailable. Acceptable values (case-insensitive): `TRUE`, `OUI`, `1`, `YES`, `VRAI`, `INDISPO`, `O`. Leave blank or `FALSE` otherwise.
        - **Dxx**: Gameweek columns (e.g., `D1` to `D38`).
            - Player ratings (e.g., `6.5`, `7.0*`). An asterisk `*` indicates a goal scored.
            - Parentheses `()` indicate a substitute appearance (e.g., `(6.0)`).
            - `0` or blank/empty cells indicate the player did not play that gameweek.
        """)

if __name__ == "__main__":
    main()

