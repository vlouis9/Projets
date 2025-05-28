import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="MPG Auction Strategist",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; /* Slightly reduced for balance */
        font-weight: bold;
        color: #004080; /* Darker blue */
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Roboto', sans-serif; /* Modern font */
    }
    .section-header {
        font-size: 1.4rem; /* Adjusted for hierarchy */
        font-weight: bold;
        color: #006847; /* Darker green */
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #006847;
        padding-bottom: 0.3rem;
    }
    .stButton>button {
        background-color: #004080; /* Match header */
        color: white;
        font-weight: bold;
        border-radius: 0.3rem; /* Softer radius */
        padding: 0.4rem 0.8rem; /* Adjusted padding */
        border: none; /* Cleaner look */
    }
    .stButton>button:hover {
        background-color: #003060; /* Darker on hover */
        color: white;
    }
    .stSlider [data-baseweb="slider"] {
        padding-bottom: 12px; 
    }
    /* Sidebar styling for better readability */
    .css-1d391kg { /* Streamlit's sidebar class, might change with versions */
        background-color: #f8f9fa; /* Light grey background */
        padding-top: 1rem;
    }
    .stExpander {
        border: 1px solid #e0e0e0;
        border-radius: 0.3rem;
        margin-bottom: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# --- Constants and Predefined Profiles ---
DEFAULT_N_RECENT_GAMES = 5
DEFAULT_MIN_RECENT_GAMES_PLAYED = 0
DEFAULT_SQUAD_SIZE = 20
DEFAULT_FORMATION = "4-4-2"

PREDEFINED_PROFILES = {
    "Custom": "custom", # Special key for custom settings
    "Balanced Value": {
        "n_recent_games": 5,
        "min_recent_games_played_filter": 1,
        "kpi_weights": {
            'GK':  {'recent_avg': 0.35, 'season_avg': 0.35, 'regularity': 0.30, 'recent_goals': 0.0, 'season_goals': 0.0},
            'DEF': {'recent_avg': 0.30, 'season_avg': 0.30, 'regularity': 0.40, 'recent_goals': 0.0, 'season_goals': 0.0},
            'MID': {'recent_avg': 0.25, 'season_avg': 0.25, 'regularity': 0.20, 'recent_goals': 0.15, 'season_goals': 0.15},
            'FWD': {'recent_avg': 0.20, 'season_avg': 0.20, 'regularity': 0.15, 'recent_goals': 0.25, 'season_goals': 0.20}
        },
        "mrb_params_per_pos": {
            'GK':  {'pos_baseline_pvs': 50, 'pos_pvs_range_for_markup': 25, 'max_markup_factor': 0.20}, # Max 20% bonus
            'DEF': {'pos_baseline_pvs': 50, 'pos_pvs_range_for_markup': 25, 'max_markup_factor': 0.25}, # Max 25% bonus
            'MID': {'pos_baseline_pvs': 55, 'pos_pvs_range_for_markup': 20, 'max_markup_factor': 0.35}, # Max 35% bonus
            'FWD': {'pos_baseline_pvs': 55, 'pos_pvs_range_for_markup': 20, 'max_markup_factor': 0.50}  # Max 50% bonus
        }
    },
    "Aggressive (High PVS Focus)": {
        "n_recent_games": 5,
        "min_recent_games_played_filter": 0,
        "kpi_weights": { # Emphasize performance metrics
            'GK':  {'recent_avg': 0.4, 'season_avg': 0.4, 'regularity': 0.2, 'recent_goals': 0.0, 'season_goals': 0.0},
            'DEF': {'recent_avg': 0.35, 'season_avg': 0.35, 'regularity': 0.3, 'recent_goals': 0.0, 'season_goals': 0.0},
            'MID': {'recent_avg': 0.3, 'season_avg': 0.3, 'regularity': 0.1, 'recent_goals': 0.15, 'season_goals': 0.15},
            'FWD': {'recent_avg': 0.3, 'season_avg': 0.3, 'regularity': 0.05, 'recent_goals': 0.2, 'season_goals': 0.15}
        },
        "mrb_params_per_pos": { # Higher markups
            'GK':  {'pos_baseline_pvs': 45, 'pos_pvs_range_for_markup': 20, 'max_markup_factor': 0.30},
            'DEF': {'pos_baseline_pvs': 45, 'pos_pvs_range_for_markup': 20, 'max_markup_factor': 0.40},
            'MID': {'pos_baseline_pvs': 50, 'pos_pvs_range_for_markup': 15, 'max_markup_factor': 0.60},
            'FWD': {'pos_baseline_pvs': 50, 'pos_pvs_range_for_markup': 15, 'max_markup_factor': 0.75}
        }
    },
     "Focus on Recent Form": {
        "n_recent_games": 3, # Shorter window
        "min_recent_games_played_filter": 1,
        "kpi_weights": {
            'GK':  {'recent_avg': 0.6, 'season_avg': 0.1, 'regularity': 0.3, 'recent_goals': 0.0, 'season_goals': 0.0},
            'DEF': {'recent_avg': 0.5, 'season_avg': 0.1, 'regularity': 0.4, 'recent_goals': 0.0, 'season_goals': 0.0},
            'MID': {'recent_avg': 0.4, 'season_avg': 0.1, 'regularity': 0.2, 'recent_goals': 0.2, 'season_goals': 0.1},
            'FWD': {'recent_avg': 0.4, 'season_avg': 0.1, 'regularity': 0.1, 'recent_goals': 0.25, 'season_goals': 0.15}
        },
        "mrb_params_per_pos": { # Standard MRB
            'GK':  {'pos_baseline_pvs': 50, 'pos_pvs_range_for_markup': 25, 'max_markup_factor': 0.20},
            'DEF': {'pos_baseline_pvs': 50, 'pos_pvs_range_for_markup': 25, 'max_markup_factor': 0.25},
            'MID': {'pos_baseline_pvs': 55, 'pos_pvs_range_for_markup': 20, 'max_markup_factor': 0.35},
            'FWD': {'pos_baseline_pvs': 55, 'pos_pvs_range_for_markup': 20, 'max_markup_factor': 0.50}
        }
    }
}

class MPGAuctionStrategist:
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
        self.squad_minimums = {"GK": 2, "DEF": 6, "MID": 6, "FWD": 4}
        self.budget = 500

    @property
    def squad_minimums_sum_val(self):
        return sum(self.squad_minimums.values())

    def simplify_position(self, position: str) -> str:
        if pd.isna(position) or str(position).strip() == '':
            return 'UNKNOWN'
        pos = str(position).upper().strip()
        if pos == 'G': return 'GK'
        elif pos in ['D', 'DL', 'DC']: return 'DEF'
        elif pos in ['M', 'MD', 'MO']: return 'MID'
        elif pos == 'A': return 'FWD'
        else: return 'UNKNOWN'

    def create_player_id(self, row) -> str:
        name = str(row.get('Joueur', '')).strip()
        simplified_pos = self.simplify_position(row.get('Poste', ''))
        club = str(row.get('Club', '')).strip()
        return f"{name}_{simplified_pos}_{club}"

    def extract_rating_and_goals(self, rating_str) -> Tuple[Optional[float], int, bool]:
        if pd.isna(rating_str) or str(rating_str).strip() == '' or str(rating_str).strip() == '0':
            return None, 0, False # DNP
        
        rating_val_str = str(rating_str).strip()
        goals = rating_val_str.count('*')
        clean_rating_str = re.sub(r'[()\*]', '', rating_val_str)
        
        try:
            rating = float(clean_rating_str)
            return rating, goals, True # Played
        except ValueError:
            return None, 0, False

    def get_gameweek_columns(self, df_columns: List[str]) -> List[str]:
        gw_cols_data = []
        for col in df_columns:
            match = re.fullmatch(r'D(\d+)', col)
            if match:
                gw_cols_data.append({'name': col, 'number': int(match.group(1))})
        sorted_gw_cols_data = sorted(gw_cols_data, key=lambda x: x['number'])
        return [col['name'] for col in sorted_gw_cols_data] # Returns D1, D2, ... D34

    def calculate_kpis(self, df: pd.DataFrame, n_recent: int) -> pd.DataFrame:
        result_df = df.copy()
        all_df_gameweek_cols_sorted = self.get_gameweek_columns(df.columns)
        
        result_df['recent_avg_rating'] = 0.0
        result_df['recent_goals'] = 0 # Integer
        result_df['season_avg_rating'] = 0.0
        result_df['season_goals'] = 0 # Integer
        result_df['recent_games_played_count'] = 0 # Integer

        for idx, row in result_df.iterrows():
            season_ratings_when_played = []
            season_goals_total = 0
            for gw_col_name in all_df_gameweek_cols_sorted:
                rating, goals, played_this_gw = self.extract_rating_and_goals(row.get(gw_col_name))
                if played_this_gw and rating is not None:
                    season_ratings_when_played.append(rating)
                    season_goals_total += goals
            
            result_df.at[idx, 'season_avg_rating'] = np.mean(season_ratings_when_played) if season_ratings_when_played else 0.0
            result_df.at[idx, 'season_goals'] = int(season_goals_total)

            recent_calendar_gws_to_check = all_df_gameweek_cols_sorted[-n_recent:]
            recent_ratings_when_played_in_window = []
            recent_goals_in_window = 0
            recent_games_played_in_window_count = 0
            for gw_col_name in recent_calendar_gws_to_check:
                rating, goals, played_this_gw = self.extract_rating_and_goals(row.get(gw_col_name))
                if played_this_gw and rating is not None:
                    recent_ratings_when_played_in_window.append(rating)
                    recent_goals_in_window += goals
                    recent_games_played_in_window_count += 1
            
            result_df.at[idx, 'recent_avg_rating'] = np.mean(recent_ratings_when_played_in_window) if recent_ratings_when_played_in_window else 0.0
            result_df.at[idx, 'recent_goals'] = int(recent_goals_in_window)
            result_df.at[idx, 'recent_games_played_count'] = int(recent_games_played_in_window_count)
        
        return result_df

    def normalize_kpis(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        result_df['norm_recent_avg'] = np.clip(result_df['recent_avg_rating'] * 10, 0, 100)
        result_df['norm_season_avg'] = np.clip(result_df['season_avg_rating'] * 10, 0, 100)
        result_df['norm_regularity'] = pd.to_numeric(result_df['%Titu'], errors='coerce').fillna(0).clip(0, 100)
        
        result_df['norm_recent_goals'] = 0.0
        result_df['norm_season_goals'] = 0.0

        for pos in ['MID', 'FWD']:
            pos_mask = result_df['simplified_position'] == pos
            if pos_mask.sum() > 0:
                result_df.loc[pos_mask, 'norm_recent_goals'] = np.clip(result_df.loc[pos_mask, 'recent_goals'] * 20, 0, 100) # 5+ goals = 100
                
                max_season_goals_pos = result_df.loc[pos_mask, 'season_goals'].max()
                if max_season_goals_pos > 0:
                    result_df.loc[pos_mask, 'norm_season_goals'] = np.clip(
                        (result_df.loc[pos_mask, 'season_goals'] / max_season_goals_pos * 100), 0, 100
                    )
        return result_df

    def calculate_pvs(self, df: pd.DataFrame, weights: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        result_df = df.copy()
        result_df['pvs'] = 0.0
        
        for pos_simplified in ['GK', 'DEF', 'MID', 'FWD']:
            pos_mask = result_df['simplified_position'] == pos_simplified
            if not pos_mask.any(): continue
            
            pos_w = weights[pos_simplified]
            current_pvs = pd.Series(0.0, index=result_df.loc[pos_mask].index)
            current_pvs += result_df.loc[pos_mask, 'norm_recent_avg'].fillna(0) * pos_w.get('recent_avg', 0)
            current_pvs += result_df.loc[pos_mask, 'norm_season_avg'].fillna(0) * pos_w.get('season_avg', 0)
            current_pvs += result_df.loc[pos_mask, 'norm_regularity'].fillna(0) * pos_w.get('regularity', 0)
            if pos_simplified in ['MID', 'FWD']:
                current_pvs += result_df.loc[pos_mask, 'norm_recent_goals'].fillna(0) * pos_w.get('recent_goals', 0)
                current_pvs += result_df.loc[pos_mask, 'norm_season_goals'].fillna(0) * pos_w.get('season_goals', 0)
            result_df.loc[pos_mask, 'pvs'] = current_pvs.clip(0, 100) # PVS is 0-100
        return result_df

    def calculate_mrb(self, df: pd.DataFrame, mrb_params_per_pos: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        result_df = df.copy()
        result_df['mrb'] = df['Cote'] # Default MRB to Cote
        
        for pos_simplified in ['GK', 'DEF', 'MID', 'FWD']:
            pos_mask = result_df['simplified_position'] == pos_simplified
            if not pos_mask.any(): continue

            params = mrb_params_per_pos[pos_simplified]
            pos_baseline_pvs = params['pos_baseline_pvs']
            pos_pvs_range_for_markup = params['pos_pvs_range_for_markup']
            max_markup_factor = params['max_markup_factor'] # e.g., 0.5 for max 50% bonus

            def calc_mrb_for_player(row):
                cote = float(row['Cote'])
                pvs = float(row['pvs']) # PVS is 0-100
                
                pvs_derived_coeff = 0.0
                if pvs > pos_baseline_pvs:
                    excess_pvs = pvs - pos_baseline_pvs
                    if pos_pvs_range_for_markup > 0:
                        pvs_derived_coeff = min(max_markup_factor, (excess_pvs / pos_pvs_range_for_markup) * max_markup_factor)
                    elif excess_pvs > 0 : # if range is 0, any excess gets max factor
                        pvs_derived_coeff = max_markup_factor
                
                calculated_mrb = cote * (1 + pvs_derived_coeff)
                capped_mrb = min(calculated_mrb, cote * 2) # Max bid is 2x Cote
                final_mrb = max(cote, round(capped_mrb)) # Ensure MRB >= Cote, and integer
                return int(final_mrb)

            result_df.loc[pos_mask, 'mrb'] = result_df.loc[pos_mask].apply(calc_mrb_for_player, axis=1)

        safe_mrb = result_df['mrb'].replace(0, np.nan).astype(float)
        result_df['value_per_cost'] = result_df['pvs'] / safe_mrb
        result_df['value_per_cost'].fillna(0, inplace=True)
        return result_df

    def select_squad(self, df: pd.DataFrame, formation_key: str, target_squad_size: int,
                     min_recent_games_played: int) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        
        if min_recent_games_played > 0:
            eligible_df = df[df['recent_games_played_count'] >= min_recent_games_played].copy()
        else:
            eligible_df = df.copy()

        if 'Indispo ?' in eligible_df.columns:
             eligible_df = eligible_df[~eligible_df['Indispo ?']].copy() # Assumes 'Indispo ?' is boolean
        
        if eligible_df.empty:
            st.warning("No eligible players after filtering. Adjust filters or check data.")
            return None, None

        eligible_df = eligible_df.drop_duplicates(subset=['player_id'])
        # Ensure MRB is integer for budget calculations
        eligible_df['mrb'] = eligible_df['mrb'].round().astype(int)


        selected_player_ids = []
        squad_selection_details = [] # List of dicts {'player_id': id, 'is_starter': bool, 'mrb_cost': cost, 'position': pos, 'pvs': pvs}
        current_budget_spent = 0
        squad_pos_counts = {pos: 0 for pos in ['GK', 'DEF', 'MID', 'FWD']}
        
        # --- Phase 1: Select Starters ---
        starters_needed_map = self.formations[formation_key].copy()
        for pos, num_to_select in starters_needed_map.items():
            candidates = eligible_df[
                (eligible_df['simplified_position'] == pos) &
                (~eligible_df['player_id'].isin(selected_player_ids))
            ].sort_values(by='pvs', ascending=False)
            
            added_count = 0
            for _, player_row in candidates.iterrows():
                if added_count >= num_to_select: break
                if current_budget_spent + player_row['mrb'] <= self.budget:
                    selected_player_ids.append(player_row['player_id'])
                    squad_selection_details.append({'player_id': player_row['player_id'], 'is_starter': True, 
                                                    'mrb_cost': player_row['mrb'], 'position': pos, 'pvs': player_row['pvs']})
                    current_budget_spent += player_row['mrb']
                    squad_pos_counts[pos] += 1
                    added_count += 1
        
        # --- Phase 2: Fulfill Overall Squad Minimums ---
        for pos, overall_min in self.squad_minimums.items():
            needed = max(0, overall_min - squad_pos_counts[pos])
            if needed == 0: continue
            candidates = eligible_df[
                (eligible_df['simplified_position'] == pos) &
                (~eligible_df['player_id'].isin(selected_player_ids))
            ].sort_values(by='value_per_cost', ascending=False)
            added_count = 0
            for _, player_row in candidates.iterrows():
                if added_count >= needed: break
                if current_budget_spent + player_row['mrb'] <= self.budget:
                    selected_player_ids.append(player_row['player_id'])
                    squad_selection_details.append({'player_id': player_row['player_id'], 'is_starter': False, 
                                                    'mrb_cost': player_row['mrb'], 'position': pos, 'pvs': player_row['pvs']})
                    current_budget_spent += player_row['mrb']
                    squad_pos_counts[pos] += 1
                    added_count += 1

        # --- Phase 3: Complete to Total Squad Size ---
        remaining_slots = max(0, target_squad_size - len(selected_player_ids))
        if remaining_slots > 0:
            candidates = eligible_df[
                (~eligible_df['player_id'].isin(selected_player_ids))
            ].sort_values(by='value_per_cost', ascending=False)
            added_count = 0
            for _, player_row in candidates.iterrows():
                if added_count >= remaining_slots: break
                if current_budget_spent + player_row['mrb'] <= self.budget:
                    selected_player_ids.append(player_row['player_id'])
                    squad_selection_details.append({'player_id': player_row['player_id'], 'is_starter': False, 
                                                    'mrb_cost': player_row['mrb'], 'position': player_row['simplified_position'], 
                                                    'pvs': player_row['pvs']})
                    current_budget_spent += player_row['mrb']
                    squad_pos_counts[player_row['simplified_position']] += 1
                    added_count += 1
        
        if not squad_selection_details:
            st.warning("Could not select any players with current settings.")
            return None, None

        # Construct final DataFrame
        final_squad_df = df[df['player_id'].isin(selected_player_ids)].copy()
        
        starter_map = {item['player_id']: item['is_starter'] for item in squad_selection_details}
        mrb_cost_map = {item['player_id']: item['mrb_cost'] for item in squad_selection_details}
        pvs_map = {item['player_id']: item['pvs'] for item in squad_selection_details} # To ensure PVS from selection is used

        final_squad_df['is_starter'] = final_squad_df['player_id'].map(starter_map)
        final_squad_df['mrb_actual_cost'] = final_squad_df['player_id'].map(mrb_cost_map).round().astype(int)
        final_squad_df['pvs_in_squad'] = final_squad_df['player_id'].map(pvs_map)


        total_squad_pvs = final_squad_df['pvs_in_squad'].sum()
        total_starters_pvs = final_squad_df[final_squad_df['is_starter']]['pvs_in_squad'].sum()

        squad_summary = {
            'total_players': len(final_squad_df),
            'total_cost': final_squad_df['mrb_actual_cost'].sum(),
            'remaining_budget': self.budget - final_squad_df['mrb_actual_cost'].sum(),
            'position_counts': final_squad_df['simplified_position'].value_counts().to_dict(),
            'total_squad_pvs': total_squad_pvs,
            'total_starters_pvs': total_starters_pvs
        }

        # Validation warnings
        for pos, min_val in self.squad_minimums.items():
            if squad_summary['position_counts'].get(pos, 0) < min_val:
                st.warning(f"Squad minimum for {pos} not met ({squad_summary['position_counts'].get(pos, 0)}/{min_val}).")
        if squad_summary['total_players'] < self.squad_minimums_sum_val:
             st.warning(f"Total players ({squad_summary['total_players']}) is less than overall minimum ({self.squad_minimums_sum_val}).")
        elif squad_summary['total_players'] < target_squad_size :
             st.warning(f"Selected {squad_summary['total_players']} players out of target {target_squad_size}.")
        return final_squad_df, squad_summary

# --- Main Streamlit App UI ---
def main():
    st.markdown('<h1 class="main-header">⚽ MPG Auction Strategist</h1>', unsafe_allow_html=True)
    
    strategist = MPGAuctionStrategist()
    
    # Initialize session state for profiles and inputs if not already present
    if 'current_profile_name' not in st.session_state:
        st.session_state.current_profile_name = "Balanced Value" # Default profile

    # --- Sidebar ---
    st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFr1EWtMR2tHq1FwHnCHqg2uXv1JMLYQlRZw&s", width=100)
    st.sidebar.markdown('<h2 class="section-header" style="margin-top:0;">⚙️ Settings & Controls</h2>', unsafe_allow_html=True)

    # Profile selection
    profile_names = list(PREDEFINED_PROFILES.keys())
    # Ensure "Custom" is an option if it was selected before
    if st.session_state.current_profile_name not in profile_names and st.session_state.current_profile_name == "Custom":
        pass # Keep "Custom" if it was the state
    elif st.session_state.current_profile_name not in profile_names: # If current state unknown, default
         st.session_state.current_profile_name = "Balanced Value"


    selected_profile_name = st.sidebar.selectbox(
        "⚙️ Select Settings Profile",
        options=profile_names,
        index=profile_names.index(st.session_state.current_profile_name),
        help="Choose a predefined settings profile or select 'Custom' to set all values manually."
    )
    
    # If profile changed, update session state for all sub-parameters
    if selected_profile_name != st.session_state.current_profile_name and selected_profile_name != "Custom":
        st.session_state.current_profile_name = selected_profile_name
        profile_values = PREDEFINED_PROFILES[selected_profile_name]
        st.session_state.n_recent = profile_values.get("n_recent_games", DEFAULT_N_RECENT_GAMES)
        st.session_state.min_recent_filter = profile_values.get("min_recent_games_played_filter", DEFAULT_MIN_RECENT_GAMES_PLAYED)
        st.session_state.kpi_weights = profile_values.get("kpi_weights", {})
        st.session_state.mrb_params_per_pos = profile_values.get("mrb_params_per_pos", {})
    elif selected_profile_name == "Custom":
         st.session_state.current_profile_name = "Custom"


    # Load current values from session state or profile defaults
    current_n_recent = st.session_state.get('n_recent', PREDEFINED_PROFILES[st.session_state.current_profile_name].get("n_recent_games", DEFAULT_N_RECENT_GAMES) if st.session_state.current_profile_name != "Custom" else DEFAULT_N_RECENT_GAMES)
    current_min_recent_filter = st.session_state.get('min_recent_filter', PREDEFINED_PROFILES[st.session_state.current_profile_name].get("min_recent_games_played_filter", DEFAULT_MIN_RECENT_GAMES_PLAYED) if st.session_state.current_profile_name != "Custom" else DEFAULT_MIN_RECENT_GAMES_PLAYED)
    current_kpi_weights = st.session_state.get('kpi_weights', PREDEFINED_PROFILES[st.session_state.current_profile_name].get("kpi_weights", {}) if st.session_state.current_profile_name != "Custom" else {})
    current_mrb_params_per_pos = st.session_state.get('mrb_params_per_pos', PREDEFINED_PROFILES[st.session_state.current_profile_name].get("mrb_params_per_pos", {}) if st.session_state.current_profile_name != "Custom" else {})


    with st.sidebar.expander("📁 File Upload & Global Data Settings", expanded=True):
        uploaded_file = st.file_uploader(
            "Upload MPG Ratings File (CSV/Excel)", type=['csv', 'xlsx', 'xls'],
            help="Columns: Joueur, Poste, Club, Cote, %Titu, Indispo? (optional), Gameweeks (D1...D34)."
        )
        n_recent_ui = st.number_input(
            "Recent Games Window (N)", min_value=1, max_value=38, value=current_n_recent,
            help="Number of most recent calendar gameweeks for 'Recent Form'. Avg rating is from games *played* within this window."
        )
        min_recent_games_filter_ui = st.number_input(
            "Filter: Min Games Played in Recent N Weeks", min_value=0, max_value=n_recent_ui, value=current_min_recent_filter,
            help=f"Exclude players who played in fewer than this in the '{n_recent_ui}' recent weeks. '0' = no filter."
        )
        # If user changes these, switch to custom profile
        if n_recent_ui != current_n_recent or min_recent_games_filter_ui != current_min_recent_filter:
            st.session_state.current_profile_name = "Custom"
        st.session_state.n_recent = n_recent_ui
        st.session_state.min_recent_filter = min_recent_games_filter_ui


    if uploaded_file is not None:
        df_input = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded_file)
        # Basic validation
        required_cols = ['Joueur', 'Poste', 'Club', 'Cote', '%Titu']
        if not all(col in df_input.columns for col in required_cols):
            st.error(f"File missing required columns: {', '.join(required_cols)}.")
            return

        df_processed = df_input.copy()
        df_processed['simplified_position'] = df_processed['Poste'].apply(strategist.simplify_position)
        df_processed['player_id'] = df_processed.apply(strategist.create_player_id, axis=1)
        df_processed['Cote'] = pd.to_numeric(df_processed['Cote'], errors='coerce').fillna(1).clip(lower=1).round().astype(int)
        if 'Indispo ?' not in df_processed.columns: df_processed['Indispo ?'] = False
        else: df_processed['Indispo ?'] = df_processed['Indispo ?'].astype(str).str.upper().isin(['TRUE', 'OUI', '1', 'YES', 'VRAI'])
        st.sidebar.success(f"✅ File '{uploaded_file.name}' loaded ({len(df_processed)} players).")

        with st.sidebar.expander("👥 Squad Building Parameters", expanded=False):
            formation_key_ui = st.selectbox("Preferred Starting Formation", options=list(strategist.formations.keys()), index=list(strategist.formations.keys()).index(st.session_state.get('formation_key',DEFAULT_FORMATION)))
            target_squad_size_ui = st.number_input("Target Total Squad Size", min_value=strategist.squad_minimums_sum_val, max_value=30, value=st.session_state.get('squad_size',DEFAULT_SQUAD_SIZE))
            if formation_key_ui != st.session_state.get('formation_key') or target_squad_size_ui != st.session_state.get('squad_size'):
                st.session_state.current_profile_name = "Custom"
            st.session_state.formation_key = formation_key_ui
            st.session_state.squad_size = target_squad_size_ui


        with st.sidebar.expander("📊 KPI Weights (0.0 to 1.0)", expanded=False):
            weights_ui = {}
            for pos in ['GK', 'DEF', 'MID', 'FWD']:
                st.markdown(f'<h4>{pos}</h4>', unsafe_allow_html=True)
                default_pos_weights = PREDEFINED_PROFILES["Balanced Value"]["kpi_weights"][pos] # Fallback defaults
                current_pos_weights = current_kpi_weights.get(pos, default_pos_weights)
                weights_ui[pos] = {
                    'recent_avg': st.slider(f"Recent Avg Rating", 0.0, 1.0, float(current_pos_weights.get('recent_avg',0.0)), 0.01, key=f"{pos}_wRA", help="Avg rating in recent N played games."),
                    'season_avg': st.slider(f"Season Avg Rating", 0.0, 1.0, float(current_pos_weights.get('season_avg',0.0)), 0.01, key=f"{pos}_wSA", help="Avg rating over season's played games."),
                    'regularity': st.slider(f"Regularity (%Titu)", 0.0, 1.0, float(current_pos_weights.get('regularity',0.0)), 0.01, key=f"{pos}_wR", help="%Titu."),
                    'recent_goals': st.slider(f"Recent Goals", 0.0, 1.0, float(current_pos_weights.get('recent_goals',0.0)) if pos in ['MID','FWD'] else 0.0, 0.01, key=f"{pos}_wRG", help="Goals in recent N played games (MID/FWD)."),
                    'season_goals': st.slider(f"Season Goals", 0.0, 1.0, float(current_pos_weights.get('season_goals',0.0)) if pos in ['MID','FWD'] else 0.0, 0.01, key=f"{pos}_wSG", help="Total season goals (MID/FWD).")
                }
            if weights_ui != current_kpi_weights: # Approximation, direct dict comparison works
                st.session_state.current_profile_name = "Custom"
            st.session_state.kpi_weights = weights_ui


        with st.sidebar.expander("💰 MRB Parameters (Per Position)", expanded=False):
            mrb_params_ui = {}
            for pos in ['GK', 'DEF', 'MID', 'FWD']:
                st.markdown(f'<h4>{pos}</h4>', unsafe_allow_html=True)
                default_pos_mrb_params = PREDEFINED_PROFILES["Balanced Value"]["mrb_params_per_pos"][pos]
                current_pos_mrb_params = current_mrb_params_per_pos.get(pos, default_pos_mrb_params)
                mrb_params_ui[pos] = {
                    'pos_baseline_pvs': st.number_input(f"PVS Baseline (0-100)", min_value=0, max_value=100, value=int(current_pos_mrb_params.get('pos_baseline_pvs',50)), step=1, key=f"{pos}_mrbBPVS", help="PVS needed to be 'worth' Cote."),
                    'max_markup_factor': st.slider(f"Max Markup Factor (Bonus %)", 0.0, 1.0, float(current_pos_mrb_params.get('max_markup_factor',0.2)), 0.01, key=f"{pos}_mrbMMF", help="Max bonus over Cote as decimal (e.g., 0.5 = 50% bonus). MRB capped at 2x Cote."),
                    'pos_pvs_range_for_markup': st.number_input(f"PVS pts for Max Markup", min_value=1, max_value=50, value=int(current_pos_mrb_params.get('pos_pvs_range_for_markup',25)), step=1, key=f"{pos}_mrbPRM", help="PVS pts above baseline for max markup.")
                }
            if mrb_params_ui != current_mrb_params_per_pos: # Approximation
                st.session_state.current_profile_name = "Custom"
            st.session_state.mrb_params_per_pos = mrb_params_ui
        
        # Global MRB cap (not per position)
        # This was in the MRB_Params dict before, now it's a global setting for the MRB calculation.
        # Let's put it with other global settings for MRB logic, or it's implied by the 2x Cote cap.
        # The formula `min(calculated_mrb, Cote * 2)` handles the per-player cap.
        # The old 'absolute_max_bid' parameter for MRB is removed as per new MRB logic.


        if st.sidebar.button("🚀 Calculate Optimal Squad & MRBs", type="primary", use_container_width=True):
            with st.spinner("🧠 Strategizing... This might take a moment!"):
                try:
                    # Use values from session state for calculations
                    n_recent_calc = st.session_state.n_recent
                    min_recent_filter_calc = st.session_state.min_recent_filter
                    weights_calc = st.session_state.kpi_weights
                    mrb_params_per_pos_calc = st.session_state.mrb_params_per_pos
                    formation_calc = st.session_state.formation_key
                    squad_size_calc = st.session_state.squad_size

                    df_kpis = strategist.calculate_kpis(df_processed, n_recent_calc)
                    df_norm_kpis = strategist.normalize_kpis(df_kpis)
                    df_pvs = strategist.calculate_pvs(df_norm_kpis, weights_calc)
                    df_mrb = strategist.calculate_mrb(df_pvs, mrb_params_per_pos_calc)
                    
                    squad_df_result, squad_summary_result = strategist.select_squad(
                        df_mrb, formation_calc, squad_size_calc, min_recent_filter_calc
                    )
                    
                    st.session_state['df_for_display'] = df_mrb
                    st.session_state['squad_df_result'] = squad_df_result
                    st.session_state['squad_summary_result'] = squad_summary_result
                    st.session_state['selected_formation_key_display'] = formation_calc # For display
                    st.success("✅ Squad calculation complete!")
                except Exception as e:
                    st.error(f"💥 Error during calculation: {str(e)}")
                    st.exception(e)

        # --- Main Panel Display Logic (largely unchanged but uses updated session state keys) ---
        if 'squad_df_result' in st.session_state and st.session_state['squad_df_result'] is not None and not st.session_state['squad_df_result'].empty:
            col_main_results, col_summary_sidebar = st.columns([3, 1])

            with col_main_results:
                st.markdown('<h2 class="section-header">🏆 Suggested Squad</h2>', unsafe_allow_html=True)
                squad_display_df = st.session_state['squad_df_result'].copy()
                # Ensure MRB, Cote, goals are int for display, others rounded
                squad_display_df['Cote'] = squad_display_df['Cote'].round().astype(int)
                squad_display_df['mrb_actual_cost'] = squad_display_df['mrb_actual_cost'].round().astype(int)
                squad_display_df['recent_goals'] = squad_display_df['recent_goals'].round().astype(int)
                squad_display_df['season_goals'] = squad_display_df['season_goals'].round().astype(int)

                squad_display_cols = ['Joueur', 'Club', 'simplified_position', 'is_starter', 
                                      'mrb_actual_cost', 'Cote', 'pvs_in_squad', 
                                      'recent_avg_rating', 'season_avg_rating', '%Titu',
                                      'recent_goals', 'season_goals', 'value_per_cost']
                display_squad_cols_exist = [col for col in squad_display_cols if col in squad_display_df.columns]
                squad_display_df = squad_display_df[display_squad_cols_exist]
                squad_display_df.rename(columns={
                    'Joueur': 'Player', 'simplified_position': 'Pos', 'is_starter': 'Starter',
                    'mrb_actual_cost': 'MRB (Cost Paid)', 'Cote': 'Listed Price', 'pvs_in_squad': 'PVS (0-100)',
                    'recent_avg_rating': 'Rec.Avg.Rate', 'season_avg_rating': 'Sea.Avg.Rate',
                    '%Titu': 'Regularity %', 'recent_goals': 'Rec.Goals', 'season_goals': 'Sea.Goals',
                    'value_per_cost': 'PVS/MRB Ratio'
                }, inplace=True)
                
                # Round other floats
                for col in ['PVS (0-100)', 'Rec.Avg.Rate', 'Sea.Avg.Rate', 'PVS/MRB Ratio']:
                    if col in squad_display_df.columns: squad_display_df[col] = squad_display_df[col].round(2)
                
                pos_order = ['GK', 'DEF', 'MID', 'FWD']
                squad_display_df['Pos'] = pd.Categorical(squad_display_df['Pos'], categories=pos_order, ordered=True)
                squad_display_df = squad_display_df.sort_values(by=['Starter', 'Pos', 'PVS (0-100)'], ascending=[False, True, False])
                st.dataframe(squad_display_df, use_container_width=True, hide_index=True)

            with col_summary_sidebar:
                st.markdown('<h2 class="section-header">📈 Squad Summary</h2>', unsafe_allow_html=True)
                summary = st.session_state['squad_summary_result']
                if summary:
                    st.metric("Budget Spent (MRB)", f"€ {summary['total_cost']:.0f} / {strategist.budget}", help=f"Remaining: € {summary['remaining_budget']:.0f}")
                    st.metric("Final Squad Size", f"{summary['total_players']} players (Target: {st.session_state.squad_size})")
                    st.metric("Total Squad PVS", f"{summary.get('total_squad_pvs',0):.2f}")
                    st.metric("Starters PVS", f"{summary.get('total_starters_pvs',0):.2f}")
                    st.info(f"**Target Formation:** {st.session_state.get('selected_formation_key_display', 'N/A')}")
                    st.markdown("**Actual Positional Breakdown:**")
                    for pos_cat in pos_order:
                        count = summary['position_counts'].get(pos_cat, 0)
                        min_req = strategist.squad_minimums.get(pos_cat,0)
                        st.write(f"• **{pos_cat}:** {count} (Min: {min_req})")
                else: st.warning("Squad summary unavailable.")
            
            st.markdown('<hr><h2 class="section-header">📋 Full Player Database & Calculated Values</h2>', unsafe_allow_html=True)
            df_full_display = st.session_state['df_for_display'].copy()
            df_full_display['Cote'] = df_full_display['Cote'].round().astype(int)
            df_full_display['mrb'] = df_full_display['mrb'].round().astype(int)
            df_full_display['recent_goals'] = df_full_display['recent_goals'].round().astype(int)
            df_full_display['season_goals'] = df_full_display['season_goals'].round().astype(int)

            # Select and rename columns for this table too
            # (Code for selecting, renaming, rounding, searching, displaying, and downloading full list as before)
            # For brevity, assuming similar column selection and renaming as in previous response block for this full table
            all_stats_cols = [
                'Joueur', 'Club', 'simplified_position', 'Poste', 'Indispo ?', 'Cote', 'pvs', 'mrb', 'value_per_cost',
                'recent_avg_rating', 'season_avg_rating', '%Titu', 'recent_goals', 'season_goals', 'recent_games_played_count',
                'norm_recent_avg', 'norm_season_avg', 'norm_regularity', 'norm_recent_goals', 'norm_season_goals'
            ]
            display_all_stats_cols_exist = [col for col in all_stats_cols if col in df_full_display.columns]
            df_full_display = df_full_display[display_all_stats_cols_exist]

            df_full_display.rename(columns={ 
                'Joueur': 'Player', 'simplified_position': 'Simp.Pos', 'Poste':'Orig.Pos', 'Indispo ?': 'Unavailable',
                'Cote': 'Listed Price', 'pvs': 'PVS', 'mrb': 'Calc. MRB', 'value_per_cost': 'Val/MRB',
                'recent_avg_rating': 'Rec.Avg.R', 'season_avg_rating': 'Sea.Avg.R',
                '%Titu': 'Reg.%', 'recent_goals': 'Rec.G', 'season_goals': 'Sea.G',
                'recent_games_played_count': 'Rec.Games Plyd',
                'norm_recent_avg': 'N.Rec.Avg', 'norm_season_avg': 'N.Sea.Avg',
                'norm_regularity': 'N.Reg.%', 'norm_recent_goals': 'N.Rec.G', 'norm_season_goals': 'N.Sea.G'
            }, inplace=True)
            
            # Round relevant columns
            for col_name in df_full_display.columns:
                if 'Price' in col_name or 'MRB' in col_name or 'Goals' in col_name or 'Plyd' in col_name : #Integers
                    if col_name in df_full_display.columns: df_full_display[col_name] = pd.to_numeric(df_full_display[col_name], errors='coerce').fillna(0).round().astype(int)
                elif 'PVS' in col_name or 'Ratio' in col_name or 'Avg' in col_name or 'Reg.%' in col_name : #Floats (2 decimal)
                    if col_name in df_full_display.columns: df_full_display[col_name] = pd.to_numeric(df_full_display[col_name], errors='coerce').fillna(0.0).round(2)


            search_term_all = st.text_input("🔍 Search All Players:", key="search_all_players_input_key")
            if search_term_all:
                search_mask_all = df_full_display.apply(lambda row: row.astype(str).str.contains(search_term_all, case=False, na=False).any(), axis=1)
                df_full_display = df_full_display[search_mask_all]
            
            st.dataframe(df_full_display.sort_values(by='PVS', ascending=False), use_container_width=True, hide_index=True, height=600)
            csv_export_all = df_full_display.to_csv(index=False).encode('utf-8')
            st.download_button(label="📥 Download Full Player Analysis (CSV)", data=csv_export_all, file_name="mpg_full_player_analysis.csv", mime="text/csv")


        elif 'df_for_display' not in st.session_state and uploaded_file:
             st.info("📊 Configure settings and click 'Calculate' to view results.")
    else:
        st.info("👈 Please upload your MPG ratings file to begin.")
        # Display Expected File Format Guide (as before)
        st.markdown('<hr><h2 class="section-header">📋 Expected File Format Guide</h2>', unsafe_allow_html=True)
        example_data = {
            'Joueur': ['Player A', 'Player B', 'Player C (Unavailable)', 'Player D'],
            'Poste': ['A', 'M', 'D', 'G'], 
            'Club': ['Club X', 'Club Y', 'Club X', 'Club Z'], 
            'Indispo ?': ['', '', 'TRUE', ''], 
            'Cote': [45, 30, 15, 10], 
            '%Titu': [90, 75, 80, 95], 
            'D34': ['7.5*', '6.5', '(5.0)', '7.0'], 
            'D33': ['(6.0)**', '7.0*', '6.0', '0'], 
            'D32': ['', '5.5', '4.5*', '(6.5)'],
        }
        st.dataframe(pd.DataFrame(example_data), use_container_width=True, hide_index=True)
        st.markdown("""**Key Column Explanations:** (As previously defined)""")


if __name__ == "__main__":
    main()
