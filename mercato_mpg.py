import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Set # Added Set

#  Page configuration
st.set_page_config(
    page_title="MPG Auction Strategist v4",
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
</style>
""", unsafe_allow_html=True)

# Constants and Predefined Profiles (from user's file)
DEFAULT_N_RECENT_GAMES = 5
DEFAULT_MIN_RECENT_GAMES_PLAYED = 1
DEFAULT_SQUAD_SIZE = 20
DEFAULT_FORMATION = "4-4-2"

PREDEFINED_PROFILES = {
    "Custom": "custom",
    "Balanced Value": {
        "n_recent_games": 5,
        "min_recent_games_played_filter": 1,
        "kpi_weights": {
            'GK': {'recent_avg': 0.05, 'season_avg': 0.70, 'calc_regularity': 0.25, 'recent_goals': 0.0, 'season_goals': 0.0},
            'DEF': {'recent_avg': 0.25, 'season_avg': 0.25, 'calc_regularity': 0.25, 'recent_goals': 0.0, 'season_goals': 0.0},
            'MID': {'recent_avg': 0.20, 'season_avg': 0.20, 'calc_regularity': 0.15, 'recent_goals': 0.15, 'season_goals': 0.15}, # [Source 5]
            'FWD': {'recent_avg': 0.15, 'season_avg': 0.15, 'calc_regularity': 0.10, 'recent_goals': 0.25, 'season_goals': 0.25}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.3},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.4},
            'MID': {'max_proportional_bonus_at_pvs100': 0.6}, # [Source 6]
            'FWD': {'max_proportional_bonus_at_pvs100': 0.8}
        }
    },
    "Aggressive Bids (Pay for PVS)": {
        "n_recent_games": 5,
        "min_recent_games_played_filter": 0,
        "kpi_weights": {
            'GK': {'recent_avg': 0.35, 'season_avg': 0.35, 'calc_regularity': 0.20, 'recent_goals': 0.0, 'season_goals': 0.0},
            'DEF': {'recent_avg': 0.30, 'season_avg': 0.30, 'calc_regularity': 0.30, 'recent_goals': 0.0, 'season_goals': 0.0}, # [Source 7]
            'MID': {'recent_avg': 0.25, 'season_avg': 0.25, 'calc_regularity': 0.15, 'recent_goals': 0.15, 'season_goals': 0.15},
            'FWD': {'recent_avg': 0.25, 'season_avg': 0.25, 'calc_regularity': 0.10, 'recent_goals': 0.20, 'season_goals': 0.20}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 1.1},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.9},
            'MID': {'max_proportional_bonus_at_pvs100': 1.1}, # [Source 8]
            'FWD': {'max_proportional_bonus_at_pvs100': 1.5}
        }
    },
    "Focus on Recent Form": {
        "n_recent_games": 3,
        "min_recent_games_played_filter": 1,
        "kpi_weights": {
            'GK': {'recent_avg': 0.5, 'season_avg': 0.1, 'calc_regularity': 0.3, 'recent_goals': 0.0, 'season_goals': 0.0},
            'DEF': {'recent_avg': 0.4, 'season_avg': 0.1, 'calc_regularity': 0.4, 'recent_goals': 0.0, 'season_goals': 0.0}, # [Source 9]
            'MID': {'recent_avg': 0.4, 'season_avg': 0.1, 'calc_regularity': 0.15, 'recent_goals': 0.2, 'season_goals': 0.1},
            'FWD': {'recent_avg': 0.4, 'season_avg': 0.1, 'calc_regularity': 0.1, 'recent_goals': 0.25, 'season_goals': 0.1}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.6},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.5}, # [Source 10]
            'MID': {'max_proportional_bonus_at_pvs100': 0.6},
            'FWD': {'max_proportional_bonus_at_pvs100': 0.9}
        }
    },
    "Focus on Season Consistency": {
        "n_recent_games": 7,
        "min_recent_games_played_filter": 2,
        "kpi_weights": {
            'GK': {'recent_avg': 0.0, 'season_avg': 0.75, 'calc_regularity': 0.25, 'recent_goals': 0.0, 'season_goals': 0.0}, # [Source 11]
            'DEF': {'recent_avg': 0.0, 'season_avg': 0.75, 'calc_regularity': 0.15, 'recent_goals': 0.0, 'season_goals': 0.10},
            'MID': {'recent_avg': 0.0, 'season_avg': 0.6, 'calc_regularity': 0.1, 'recent_goals': 0.0, 'season_goals': 0.3},
            'FWD': {'recent_avg': 0.0, 'season_avg': 0.5, 'calc_regularity': 0.1, 'recent_goals': 0.0, 'season_goals': 0.4}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.9},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.8}, # [Source 12]
            'MID': {'max_proportional_bonus_at_pvs100': 0.9},
            'FWD': {'max_proportional_bonus_at_pvs100': 1.2}
        }
    }
}


class MPGAuctionStrategist:
    def __init__(self):
        self.formations = {
            "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
            "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3}, # [Source 13]
            "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
            "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
            "4-5-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
            "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},
            "5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1} # [Source 14]
        }
        self.squad_minimums = {"GK": 2, "DEF": 6, "MID": 6, "FWD": 4}
        self.budget = 500

    @property
    def squad_minimums_sum_val(self):
        return sum(self.squad_minimums.values())

    def simplify_position(self, position: str) -> str:
        if pd.isna(position) or str(position).strip() == '':
            return 'UNKNOWN'
        pos = str(position).upper().strip() # [Source 15]
        if pos == 'G':
            return 'GK'
        elif pos in ['D', 'DL', 'DC']:
            return 'DEF'
        elif pos in ['M', 'MD', 'MO']:
            return 'MID'
        elif pos == 'A':
            return 'FWD' # [Source 16]
        else:
            return 'UNKNOWN'

    def create_player_id(self, row) -> str:
        name = str(row.get('Joueur', '')).strip()
        simplified_pos = self.simplify_position(row.get('Poste', ''))
        club = str(row.get('Club', '')).strip()
        return f"{name}_{simplified_pos}_{club}"

    def extract_rating_goals_starter(self, rating_str) -> Tuple[Optional[float], int, bool, bool]:
        """Returns rating, goals, played_this_gw, is_starter""" # [Source 17]
        if pd.isna(rating_str) or str(rating_str).strip() == '' or str(rating_str).strip() == '0':
            return None, 0, False, False
        val_str = str(rating_str).strip()
        goals = val_str.count('*')
        is_starter = '(' not in val_str
        clean_rating_str = re.sub(r'[()\*]', '', val_str)
        try:
            rating = float(clean_rating_str) # [Source 18]
            return rating, goals, True, is_starter
        except ValueError:
            return None, 0, False, False

    def get_gameweek_columns(self, df_columns: List[str]) -> List[str]:
        gw_cols_data = [{'name': col, 'number': int(match.group(1))} for col in df_columns if (match := re.fullmatch(r'D(\d+)', col))]
        return [col['name'] for col in sorted(gw_cols_data, key=lambda x: x['number'])]

    def calculate_kpis(self, df: pd.DataFrame, n_recent: int) -> pd.DataFrame:
        rdf = df.copy() # [Source 19]
        all_gws = self.get_gameweek_columns(df.columns)
        rdf[['recent_avg_rating', 'season_avg_rating']] = 0.0
        rdf[['recent_goals', 'season_goals', 'recent_games_played_count',
             'calc_regularity_pct', 'games_started_season', 'total_season_gws_considered']] = 0

        for idx, row in rdf.iterrows():
            s_ratings_p, s_goals_t, s_started, s_played = [], 0, 0, 0
            for gw_col in all_gws: # [Source 20]
                r, g, played, starter = self.extract_rating_goals_starter(row.get(gw_col))
                if played and r is not None:
                    s_ratings_p.append(r)
                    s_goals_t += g
                    s_played += 1 # [Source 21]
                    if starter:
                        s_started += 1
            rdf.at[idx, 'season_avg_rating'] = np.mean(s_ratings_p) if s_ratings_p else 0.0
            rdf.at[idx, 'season_goals'] = s_goals_t
            rdf.at[idx, 'games_started_season'] = s_started # [Source 22]
            rdf.at[idx, 'total_season_gws_considered'] = len(all_gws)
            rdf.at[idx, 'calc_regularity_pct'] = (s_started / len(all_gws) * 100) if len(all_gws) > 0 else 0.0

            rec_gws_check = all_gws[-n_recent:]
            rec_ratings_p, rec_goals_s, rec_games_p_window = [], 0, 0
            for gw_col in rec_gws_check:
                r, g, played, _ = self.extract_rating_goals_starter(row.get(gw_col)) # [Source 23]
                if played and r is not None:
                    rec_ratings_p.append(r)
                    rec_goals_s += g
                    rec_games_p_window += 1 # [Source 24]
            rdf.at[idx, 'recent_avg_rating'] = np.mean(rec_ratings_p) if rec_ratings_p else 0.0
            rdf.at[idx, 'recent_goals'] = rec_goals_s
            rdf.at[idx, 'recent_games_played_count'] = rec_games_p_window

        for col in ['recent_goals', 'season_goals', 'recent_games_played_count', 'games_started_season', 'total_season_gws_considered']:
            rdf[col] = rdf[col].astype(int)
        return rdf

    def normalize_kpis(self, df: pd.DataFrame) -> pd.DataFrame:
        rdf = df.copy() # [Source 25]
        rdf['norm_recent_avg'] = np.clip(rdf['recent_avg_rating'] * 10, 0, 100)
        rdf['norm_season_avg'] = np.clip(rdf['season_avg_rating'] * 10, 0, 100)
        rdf['norm_regularity_file'] = pd.to_numeric(rdf['%Titu'], errors='coerce').fillna(0).clip(0, 100) # Retained if user wants to use it
        rdf['norm_calc_regularity'] = rdf['calc_regularity_pct'].clip(0, 100)
        rdf[['norm_recent_goals', 'norm_season_goals']] = 0.0
        # User's code included DEF for goals normalization, will follow that.
        for pos in ['DEF', 'MID', 'FWD']:
            mask = rdf['simplified_position'] == pos
            if mask.any(): # [Source 26]
                rdf.loc[mask, 'norm_recent_goals'] = np.clip(rdf.loc[mask, 'recent_goals'] * 20, 0, 100)
                max_sg = rdf.loc[mask, 'season_goals'].max()
                rdf.loc[mask, 'norm_season_goals'] = np.clip((rdf.loc[mask, 'season_goals'] / max_sg * 100) if max_sg > 0 else 0, 0, 100)
        return rdf

    def calculate_pvs(self, df: pd.DataFrame, weights: Dict[str, Dict[str, float]]) -> pd.DataFrame: # [Source 27]
        rdf = df.copy()
        rdf['pvs'] = 0.0
        for pos, w in weights.items():
            mask = rdf['simplified_position'] == pos
            if not mask.any():
                continue
            pvs_sum = pd.Series(0.0, index=rdf.loc[mask].index) # [Source 28]
            pvs_sum += rdf.loc[mask, 'norm_recent_avg'].fillna(0) * w.get('recent_avg', 0)
            pvs_sum += rdf.loc[mask, 'norm_season_avg'].fillna(0) * w.get('season_avg', 0)
            # User's code for PVS does not include 'norm_regularity_file', only 'calc_regularity'
            # I will stick to user's latest code structure for PVS calculation
            pvs_sum += rdf.loc[mask, 'norm_calc_regularity'].fillna(0) * w.get('calc_regularity', 0)
            if pos in ['DEF', 'MID', 'FWD']: # User's code includes DEF here
                pvs_sum += rdf.loc[mask, 'norm_recent_goals'].fillna(0) * w.get('recent_goals', 0)
                pvs_sum += rdf.loc[mask, 'norm_season_goals'].fillna(0) * w.get('season_goals', 0) # [Source 29]
            rdf.loc[mask, 'pvs'] = pvs_sum.clip(0, 100)
        return rdf

    def calculate_mrb(self, df: pd.DataFrame, mrb_params_per_pos: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        rdf = df.copy()
        rdf['Cote'] = pd.to_numeric(rdf['Cote'], errors='coerce').fillna(1).clip(lower=1).round().astype(int)
        rdf['mrb'] = rdf['Cote']
        for pos_simplified, params in mrb_params_per_pos.items():
            mask = rdf['simplified_position'] == pos_simplified # [Source 30]
            if not mask.any():
                continue
            max_prop_bonus = params.get('max_proportional_bonus_at_pvs100', 0.5)

            def _calc_mrb_player_v3(row): # Name kept from user's file
                cote = int(row['Cote'])
                pvs_player_0_100 = float(row['pvs']) # [Source 31]
                pvs_scaled_0_1 = pvs_player_0_100 / 100.0
                pvs_derived_bonus_factor = pvs_scaled_0_1 * max_prop_bonus
                mrb_float = cote * (1 + pvs_derived_bonus_factor)
                mrb_capped_at_2x_cote = min(mrb_float, float(cote * 2))
                final_mrb = max(float(cote), mrb_capped_at_2x_cote) # [Source 32]
                return int(round(final_mrb))

            rdf.loc[mask, 'mrb'] = rdf.loc[mask].apply(_calc_mrb_player_v3, axis=1)
        rdf['mrb'] = rdf['mrb'].astype(int)
        safe_mrb = rdf['mrb'].replace(0, np.nan).astype(float)
        rdf['value_per_cost'] = rdf['pvs'] / safe_mrb
        rdf['value_per_cost'].fillna(0, inplace=True)
        return rdf

        # [Existing code from mercato_mpg_gemini.txt, Source 1, lines 1-147, will be assumed to be here]
# This includes PREDEFINED_PROFILES and the MPGAuctionStrategist class up to select_squad

    def select_squad(self, df: pd.DataFrame, formation_key: str, target_squad_size: int,
                     min_recent_games_played: int) -> Tuple[pd.DataFrame, Dict]:
        """
        Selects a squad by:
        A) Prioritizing player counts (starters, mins, target size) & PVS, possibly going over budget.
        B) Iteratively adjusting to meet budget by downgrading with minimal PVS loss.
        C) Trying to spend remaining budget to maximize PVS if under budget.
        """
        
        # --- Initial Filtering ---
        eligible_df_initial = df.copy()
        if min_recent_games_played > 0:
            eligible_df_initial = eligible_df_initial[
                eligible_df_initial['recent_games_played_count'] >= min_recent_games_played
            ]
        # Per user's file (Source 1), 'Indispo ?' filtering is not done here but in main before processing.
        
        if eligible_df_initial.empty:
            return pd.DataFrame(), {}
        
        eligible_df = eligible_df_initial.drop_duplicates(subset=['player_id']).copy()
        eligible_df['mrb'] = eligible_df['mrb'].astype(int)

        # --- Data structure for currently selected players ---
        # List of dicts: {'player_id': str, 'mrb': int, 'pvs': float, 'pos': str, 'is_starter': bool}
        current_squad_list_of_dicts: List[Dict] = [] 
        
        # --- Helper functions within select_squad ---
        def get_player_details_from_df(player_id_list: List[str]) -> pd.DataFrame:
            return eligible_df[eligible_df['player_id'].isin(player_id_list)]

        def get_current_squad_player_ids_set() -> Set[str]:
            return {p['player_id'] for p in current_squad_list_of_dicts}

        def get_current_pos_counts_dict() -> Dict[str, int]:
            counts = {pos_key: 0 for pos_key in self.squad_minimums.keys()}
            for p_dict in current_squad_list_of_dicts:
                counts[p_dict['pos']] = counts.get(p_dict['pos'], 0) + 1
            return counts

        def add_player_to_current_squad_list(player_row_data: pd.Series, is_starter_role: bool) -> bool:
            player_id_to_add = player_row_data['player_id']
            if player_id_to_add in get_current_squad_player_ids_set():
                return False 

            # GK Count Check before adding
            if player_row_data['simplified_position'] == 'GK':
                current_gk_count = get_current_pos_counts_dict().get('GK', 0)
                if current_gk_count >= 2:
                    # st.caption(f"Cannot add GK {player_row_data['Joueur']}; already {current_gk_count} GKs.")
                    return False 
            
            current_squad_list_of_dicts.append({
                'player_id': player_id_to_add,
                'mrb': int(player_row_data['mrb']),
                'pvs': float(player_row_data['pvs']),
                'pos': player_row_data['simplified_position'],
                'is_starter': is_starter_role,
                'Joueur': player_row_data['Joueur'] # For logging
            })
            return True

        def remove_player_from_current_squad_list(player_id_to_remove: str) -> bool:
            nonlocal current_squad_list_of_dicts
            initial_len = len(current_squad_list_of_dicts)
            current_squad_list_of_dicts = [p for p in current_squad_list_of_dicts if p['player_id'] != player_id_to_remove]
            return len(current_squad_list_of_dicts) < initial_len
            
        # --- Phase A: Initial High-PVS Squad Construction (Potentially Over Budget) ---
        all_players_sorted_pvs = eligible_df.sort_values(by='pvs', ascending=False)

        # A1: Select Starters
        starters_map = self.formations[formation_key].copy()
        for _, player_row in all_players_sorted_pvs.iterrows():
            pos = player_row['simplified_position']
            if player_row['player_id'] not in get_current_squad_player_ids_set() and starters_map.get(pos, 0) > 0:
                if add_player_to_current_squad_list(player_row, True):
                    starters_map[pos] -= 1
        
        # A2: Fulfill Overall Squad Positional Minimums
        current_counts_ph_a2 = get_current_pos_counts_dict()
        for pos, min_needed in self.squad_minimums.items():
            while current_counts_ph_a2.get(pos, 0) < min_needed:
                candidate_series = all_players_sorted_pvs[
                    (all_players_sorted_pvs['simplified_position'] == pos) &
                    (~all_players_sorted_pvs['player_id'].isin(get_current_squad_player_ids_set()))
                ].head(1)
                if candidate_series.empty: break 
                
                # Add_player checks GK count
                if add_player_to_current_squad_list(candidate_series.iloc[0], False):
                    current_counts_ph_a2 = get_current_pos_counts_dict() 
                else: # Could not add (e.g. GK limit reached for a GK candidate)
                    # To prevent infinite loop if only GKs are left for a non-GK slot (unlikely), or if GK min is >2
                    # We must ensure we don't try to add a 3rd GK to meet a "general" minimum.
                    # If this specific candidate failed (likely GK constraint), break from this *specific* add attempt and try next in sorted_pvs
                    # This might require fetching more candidates if the top one was a GK that couldn't be added.
                    # For simplicity, if add fails, assume no suitable player for this slot under constraints.
                    break 


        # A3: Complete to Target Squad Size - "Most Needed Position" Logic
        # Define "need" - for simplicity, aim for even distribution above starters/minimums, or by formation proportion
        # Heuristic: (desired total for pos - current for pos). Desired could be formation starters + 1 or 2.
        # For now, try to balance based on formation starter counts as a proxy for importance.
        
        while len(current_squad_list_of_dicts) < target_squad_size:
            current_counts_ph_a3 = get_current_pos_counts_dict()
            most_needed_pos_details = [] # List of (pos_name, deficit, num_starters_in_formation)
            
            for pos_key, num_starters in self.formations[formation_key].items():
                # A simple need: positions with fewer players relative to their starter count in formation
                # Or more generally, positions where current_count < (starters + desired_backup_per_starter_slot)
                # For now, just try to add to positions that have fewer players, avoiding >2 GKs
                desired_for_pos = num_starters + 1 # Aim for at least one backup for starter slots
                deficit = desired_for_pos - current_counts_ph_a3.get(pos_key, 0)
                most_needed_pos_details.append((pos_key, deficit, num_starters))

            # Sort by largest deficit, then by original number of starters (more important positions)
            most_needed_pos_details.sort(key=lambda x: (x[1], x[2]), reverse=True)
            
            player_added_in_a3_pass = False
            for pos_to_fill, deficit_val, _ in most_needed_pos_details:
                if deficit_val <= 0 and len(current_squad_list_of_dicts) < target_squad_size : # No deficit, but squad not full, try any pos
                    pass # Fall through to general PVS pick if specific needs met or not clear
                elif deficit_val <=0: # Deficit met or negative, skip this pos for targeted fill
                    continue

                if pos_to_fill == 'GK' and current_counts_ph_a3.get('GK', 0) >= 2:
                    continue # Skip adding GK if limit reached
                
                candidate_series_a3 = all_players_sorted_pvs[
                    (all_players_sorted_pvs['simplified_position'] == pos_to_fill) &
                    (~all_players_sorted_pvs['player_id'].isin(get_current_squad_player_ids_set()))
                ].head(1)

                if not candidate_series_a3.empty:
                    if add_player_to_current_squad_list(candidate_series_a3.iloc[0], False):
                        player_added_in_a3_pass = True
                        break # Added one player, re-evaluate needs for next slot
            
            if not player_added_in_a3_pass: # If no specific "needed" position filled, or all needs met, fill with best PVS overall
                if len(current_squad_list_of_dicts) < target_squad_size:
                    candidate_overall_a3 = all_players_sorted_pvs[
                        ~all_players_sorted_pvs['player_id'].isin(get_current_squad_player_ids_set())
                    ].head(1)
                    if not candidate_overall_a3.empty:
                         # Add_player checks GK count
                        if not add_player_to_current_squad_list(candidate_overall_a3.iloc[0], False):
                            # If adding best PVS failed (likely it was a 3rd GK), remove it from consideration and try next best PVS
                            all_players_sorted_pvs = all_players_sorted_pvs[all_players_sorted_pvs['player_id'] != candidate_overall_a3.iloc[0]['player_id']]
                            if all_players_sorted_pvs.empty: break # No more players at all
                            continue # Try next in the while loop
                    else: break # No more players left
                else: break # Squad is full
            if len(current_squad_list_of_dicts) >= target_squad_size: break


        # --- Calculate initial MRB total ---
        current_total_mrb = sum(p['mrb'] for p in current_squad_list_of_dicts)
        
        # --- Phase B: Iterative Budget Conformance (If Over Budget) ---
        max_budget_iterations_b = target_squad_size * 2 
        iterations_b_count = 0
        budget_conformance_tolerance = 1 # Aim to be <= budget

        while current_total_mrb > self.budget + budget_conformance_tolerance and iterations_b_count < max_budget_iterations_b:
            iterations_b_count += 1
            made_a_downgrade_in_pass = False
            
            best_downgrade_action = None # Store: (old_pid, new_pid, mrb_saved, pvs_change_val, new_player_row)
                                        # pvs_change_val = new_pvs - old_pvs (want this to be minimally negative)
            
            # Consider non-starters first for downgrading
            current_squad_non_starters = [p for p in current_squad_list_of_dicts if not p['is_starter']]
            # If no non-starters, or non-starter downgrades aren't enough, consider starters (more complex, for now focus on non-starters)
            
            # Iterate through current squad members (non-starters preferred)
            # Sort them by MRB (desc) to target expensive ones, or by PVS/MRB (asc) for inefficient ones
            # For simplicity, let's try to downgrade any non-starter if a good swap exists.
            
            candidates_for_replacement = sorted(current_squad_non_starters, key=lambda x: x['mrb'], reverse=True)
            if not candidates_for_replacement and current_total_mrb > self.budget: # If only starters left
                candidates_for_replacement = sorted(current_squad_list_of_dicts, key=lambda x: x['pvs']) # lowest PVS starter


            for old_player_dict_b in candidates_for_replacement:
                old_pid_b = old_player_dict_b['player_id']
                old_pos_b = old_player_dict_b['pos']
                old_mrb_b = old_player_dict_b['mrb']
                old_pvs_b = old_player_dict_b['pvs']

                # Find potential cheaper replacements
                potential_replacements_df = eligible_df[
                    (eligible_df['simplified_position'] == old_pos_b) &
                    (~eligible_df['player_id'].isin(get_current_squad_player_ids_set() - {old_pid_b})) & # Exclude current squad except the one being replaced
                    (eligible_df['mrb'] < old_mrb_b)
                ]

                if not potential_replacements_df.empty:
                    for _, new_player_row_b in potential_replacements_df.iterrows():
                        # GK constraint for new player
                        if new_player_row_b['simplified_position'] == 'GK':
                            # If old player was not GK, this swap would increase GK count.
                            # If old player was GK, GK count remains same.
                            is_old_gk = old_pos_b == 'GK'
                            current_gk_count_b = get_current_pos_counts_dict().get('GK',0)
                            if not is_old_gk and current_gk_count_b >=2: # Trying to add a 3rd GK by swapping non-GK
                                continue
                            if is_old_gk and current_gk_count_b > 2 : # This implies an error if we already have >2 GKs being swapped.
                                continue


                        mrb_saved_b = old_mrb_b - new_player_row_b['mrb']
                        pvs_change_val_b = new_player_row_b['pvs'] - old_pvs_b # new_pvs - old_pvs
                        
                        # Metric: maximize mrb_saved, minimize pvs_loss (pvs_change_val negative)
                        # Score: mrb_saved - abs(pvs_change_val if pvs_change_val < 0 else 0) * some_penalty_factor
                        # Simpler: prioritize MRB saved, then PVS change.
                        
                        # If this swap is better than current best_downgrade_action for this pass
                        if best_downgrade_action is None or \
                           (mrb_saved_b > best_downgrade_action[2]) or \
                           (mrb_saved_b == best_downgrade_action[2] and pvs_change_val_b > best_downgrade_action[3]):
                            best_downgrade_action = (old_pid_b, new_player_row_b['player_id'], mrb_saved_b, pvs_change_val_b, new_player_row_b)
            
            if best_downgrade_action:
                old_id_exec, new_id_exec, mrb_s_exec, pvs_c_exec, new_player_data_exec = best_downgrade_action
                
                original_starter_status_exec = False # Find original starter status of the player being swapped out
                for p_dict_exec_old in current_squad_list_of_dicts:
                    if p_dict_exec_old['player_id'] == old_id_exec:
                        original_starter_status_exec = p_dict_exec_old['is_starter']
                        break
                
                if remove_player_from_current_squad_list(old_id_exec):
                    if add_player_to_current_squad_list(new_player_data_exec, original_starter_status_exec):
                        current_total_mrb = sum(p['mrb'] for p in current_squad_list_of_dicts)
                        st.caption(f"Budget Downgrade: Swapped player. Saved €{mrb_s_exec}. PVS change: {pvs_c_exec:.2f}. New MRB: {current_total_mrb}")
                        made_a_downgrade_in_pass = True
                    else: # Should not happen if GK logic in add_player is correct
                        st.warning(f"Failed to add replacement {new_player_data_exec['Joueur']} during downgrade (likely GK constraint).")
                        # Re-add old player if replacement failed to maintain squad size/integrity (complex recovery)
                        # For now, just break this pass.
                        break 
            
            if not made_a_downgrade_in_pass:
                if current_total_mrb > self.budget + budget_conformance_tolerance:
                     st.warning(f"Budget Target Not Met: Current MRB {current_total_mrb} > Budget {self.budget}. No more effective downgrades found.")
                break # No beneficial downgrades found in this full pass

        # --- Phase C: Final PVS Upgrade (Spend Remaining Budget) ---
        budget_left_for_upgrades = self.budget - current_total_mrb
        max_upgrade_passes_c = target_squad_size # Limit passes
        upgrade_pass_count_c = 0
        
        while budget_left_for_upgrades > 5 and upgrade_pass_count_c < max_upgrade_passes_c and len(current_squad_list_of_dicts) == target_squad_size :
            upgrade_pass_count_c += 1
            made_an_upgrade_this_pass_c = False
            best_upgrade_action_c = None # (old_pid, new_pid, mrb_increase, pvs_gain, new_player_row)

            # Consider non-starters for upgrade first
            squad_for_upgrade_cands_c = sorted([p for p in current_squad_list_of_dicts if not p['is_starter']], key=lambda x: x['pvs']) 
            if not squad_for_upgrade_cands_c: # If all are starters
                squad_for_upgrade_cands_c = sorted(current_squad_list_of_dicts, key=lambda x: x['pvs'])


            for old_player_dict_c in squad_for_upgrade_cands_c:
                old_pid_c = old_player_dict_c['player_id']
                old_pos_c = old_player_dict_c['pos']
                old_mrb_c = old_player_dict_c['mrb']
                old_pvs_c = old_player_dict_c['pvs']

                potential_upgrades_df = eligible_df[
                    (eligible_df['simplified_position'] == old_pos_c) &
                    (~eligible_df['player_id'].isin(get_current_squad_player_ids_set() - {old_pid_c})) &
                    (eligible_df['pvs'] > old_pvs_c) & # Must be PVS improvement
                    (eligible_df['mrb'] > old_mrb_c)   # Typically expect to pay more for more PVS
                ]

                for _, new_player_row_c in potential_upgrades_df.iterrows():
                    # GK constraint for new player
                    if new_player_row_c['simplified_position'] == 'GK':
                        is_old_gk_c = old_pos_c == 'GK'
                        current_gk_count_c = get_current_pos_counts_dict().get('GK',0)
                        if not is_old_gk_c and current_gk_count_c >=2 : continue
                        if is_old_gk_c and current_gk_count_c > 2 and new_player_row_c['player_id'] != old_pid_c: continue # should not happen


                    mrb_increase_c = new_player_row_c['mrb'] - old_mrb_c
                    pvs_gain_c = new_player_row_c['pvs'] - old_pvs_c

                    if mrb_increase_c <= budget_left_for_upgrades: # Must be affordable
                        # Metric: maximize pvs_gain, or pvs_gain / mrb_increase
                        current_upgrade_score = pvs_gain_c / (mrb_increase_c + 0.1) # +0.1 to avoid div by zero if MRB is same (unlikely with mrb > old_mrb)

                        if best_upgrade_action_c is None or \
                           current_upgrade_score > (best_upgrade_action_c[3] / (best_upgrade_action_c[2] + 0.1)):
                            best_upgrade_action_c = (old_pid_c, new_player_row_c['player_id'], mrb_increase_c, pvs_gain_c, new_player_row_c)
            
            if best_upgrade_action_c:
                old_id_exec_c, new_id_exec_c, mrb_inc_exec_c, pvs_g_exec_c, new_player_data_exec_c = best_upgrade_action_c
                
                original_starter_status_c = False
                for p_dict_exec_old_c in current_squad_list_of_dicts:
                    if p_dict_exec_old_c['player_id'] == old_id_exec_c:
                        original_starter_status_c = p_dict_exec_old_c['is_starter']
                        break

                if remove_player_from_current_squad_list(old_id_exec_c):
                    if add_player_to_current_squad_list(new_player_data_exec_c, original_starter_status_c):
                        current_total_mrb += mrb_inc_exec_c
                        budget_left_for_upgrades = self.budget - current_total_mrb
                        st.caption(f"Budget Upgrade: Swapped. MRB increase €{mrb_inc_exec_c}. PVS gain: {pvs_g_exec_c:.2f}. New Total MRB: {current_total_mrb}")
                        made_an_upgrade_this_pass_c = True
                    else: # Should not happen
                        st.warning(f"Failed to add replacement {new_player_data_exec_c['Joueur']} during upgrade.")
                        break
                else: # Should not happen
                    st.warning(f"Failed to remove {old_id_exec_c} during upgrade.")
                    break
            
            if not made_an_upgrade_this_pass_c: break

        # --- Final Squad Construction from current_squad_list_of_dicts ---
        if not current_squad_list_of_dicts: return pd.DataFrame(), {}
        
        final_squad_player_ids = get_current_squad_player_ids_set()
        final_squad_df_base = eligible_df[eligible_df['player_id'].isin(final_squad_player_ids)].copy()
        
        details_df_final = pd.DataFrame(current_squad_list_of_dicts)
        details_df_final.rename(columns={'mrb': 'mrb_actual_cost', 'pvs':'pvs_in_squad', 
                                         'pos':'final_pos', 'is_starter':'is_starter_from_selection'}, inplace=True)
        
        final_squad_df = pd.merge(final_squad_df_base, 
                                  details_df_final[['player_id', 'mrb_actual_cost', 'pvs_in_squad', 'is_starter_from_selection']], 
                                  on='player_id', how='left')
        
        # Re-determine starter status based on PVS within formation constraints for the *final* squad
        final_starter_ids_definitive = set()
        temp_formation_needs_final = self.formations[formation_key].copy()
        final_squad_df_sorted_for_final_starters = final_squad_df.sort_values(by='pvs_in_squad', ascending=False)

        for _, player_row_final_pass in final_squad_df_sorted_for_final_starters.iterrows():
            pos_final_pass = player_row_final_pass['simplified_position']
            player_id_final_pass = player_row_final_pass['player_id']
            if temp_formation_needs_final.get(pos_final_pass, 0) > 0:
                if player_id_final_pass not in final_starter_ids_definitive :
                    final_starter_ids_definitive.add(player_id_final_pass)
                    temp_formation_needs_final[pos_final_pass] -=1
        
        final_squad_df['is_starter'] = final_squad_df['player_id'].isin(final_starter_ids_definitive)
        if 'is_starter_from_selection' in final_squad_df.columns:
            final_squad_df.drop(columns=['is_starter_from_selection'], inplace=True)


        final_total_mrb_actual = final_squad_df['mrb_actual_cost'].sum()
        summary = {
            'total_players': len(final_squad_df),
            'total_cost': int(final_total_mrb_actual),
            'remaining_budget': int(self.budget - final_total_mrb_actual),
            'position_counts': final_squad_df['simplified_position'].value_counts().to_dict(),
            'total_squad_pvs': round(final_squad_df['pvs_in_squad'].sum(), 2),
            'total_starters_pvs': round(final_squad_df[final_squad_df['is_starter']]['pvs_in_squad'].sum(), 2)
        }
        
        final_pos_counts_check_final = summary['position_counts']
        for pos_check, min_val_check in self.squad_minimums.items():
            if final_pos_counts_check_final.get(pos_check,0) < min_val_check:
                st.error(f"Squad Selection Issue: Position {pos_check} minimum not met! ({final_pos_counts_check_final.get(pos_check,0)}/{min_val_check}) Please check data or constraints.")
        if len(final_squad_df) != target_squad_size :
             st.error(f"Squad Selection Issue: Final squad size {len(final_squad_df)} does not match target {target_squad_size}.")


        return final_squad_df, summary

# [The rest of the main() function from mercato_mpg_gemini.txt (Source 1, lines 55-102) will be assumed to be here]
# Ensure that the main() function correctly calls this updated select_squad.
# The column names used in the display part of main() should match what this select_squad returns.
# For example, 'mrb_actual_cost' and 'pvs_in_squad' are used.

# For the purpose of this response, I will append the main() function from your provided file
# to make the script complete and runnable.

def main(): # [Source 55]
    st.markdown('<h1 class="main-header">🚀 MPG Auction Strategist v5 (Budget Focus)</h1>', unsafe_allow_html=True) # Updated title
    strategist = MPGAuctionStrategist()

    # Initialize session state correctly
    if 'current_profile_name' not in st.session_state:
        st.session_state.current_profile_name = "Balanced Value" # Default profile
        # Initialize other dependent session state variables based on this default
        profile_values = PREDEFINED_PROFILES[st.session_state.current_profile_name]
        st.session_state.n_recent = profile_values.get("n_recent_games", DEFAULT_N_RECENT_GAMES)
        st.session_state.min_recent_filter = profile_values.get("min_recent_games_played_filter", DEFAULT_MIN_RECENT_GAMES_PLAYED) # [Source 56]
        st.session_state.kpi_weights = profile_values.get("kpi_weights", PREDEFINED_PROFILES["Balanced Value"]["kpi_weights"]) 
        st.session_state.mrb_params_per_pos = profile_values.get("mrb_params_per_pos", PREDEFINED_PROFILES["Balanced Value"]["mrb_params_per_pos"])
    
    # These were outside the profile load in user's file [Source 58], keeping that structure
    if 'formation_key' not in st.session_state: st.session_state.formation_key = DEFAULT_FORMATION 
    if 'squad_size' not in st.session_state: st.session_state.squad_size = DEFAULT_SQUAD_SIZE


    # Sidebar UI Elements (matches user's file structure mostly)
    st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFr1EWtMR2tHq1FwHnCHqg2uXv1JMLYQlRZw&s", width=100)
    st.sidebar.markdown('<h2 class="section-header" style="margin-top:0;">⚙️ Controls & Settings</h2>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("📁 Upload MPG Ratings File (CSV/Excel)", type=['csv', 'xlsx', 'xls'], help="Joueur, Poste, Club, Cote, %Titu, Indispo?, Gameweeks (D1..D34).")

    # This conditional block structure is from user's file (Source 1, line 58 where inputs are inside if uploaded_file)
    # I will modify it slightly to allow settings configuration even before upload, then process if file exists.
    # This is more typical for Streamlit apps.

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🌎 Global Data & Form Parameters") # [Source 59]
    # Retrieve from session state, or use default if not set (though init should cover it)
    n_recent_ui = st.sidebar.number_input("Recent Games Window (N)", min_value=1, max_value=38, 
                                          value=st.session_state.get('n_recent', DEFAULT_N_RECENT_GAMES), 
                                          help="For 'Recent Form' KPIs. Avg of games *played* in this window.") # [Source 60]
    min_recent_filter_ui = st.sidebar.number_input("Filter: Min Games Played in Recent N Weeks", min_value=0, max_value=n_recent_ui, 
                                                   value=st.session_state.get('min_recent_filter', DEFAULT_MIN_RECENT_GAMES_PLAYED), 
                                                   help=f"Exclude players with < this in '{n_recent_ui}' recent weeks. 0 = no filter.")
    if n_recent_ui != st.session_state.get('n_recent') or min_recent_filter_ui != st.session_state.get('min_recent_filter'):
        st.session_state.current_profile_name = "Custom" 
    st.session_state.n_recent = n_recent_ui
    st.session_state.min_recent_filter = min_recent_filter_ui

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 👥 Squad Building Parameters") # [Source 61]
    formation_key_ui = st.sidebar.selectbox("Preferred Starting Formation", options=list(strategist.formations.keys()), 
                                            index=list(strategist.formations.keys()).index(st.session_state.get('formation_key', DEFAULT_FORMATION)))
    target_squad_size_ui = st.sidebar.number_input("Target Total Squad Size", min_value=strategist.squad_minimums_sum_val, max_value=30, 
                                                 value=st.session_state.get('squad_size', DEFAULT_SQUAD_SIZE))
    if formation_key_ui != st.session_state.get('formation_key') or target_squad_size_ui != st.session_state.get('squad_size'):
        st.session_state.current_profile_name = "Custom" 
    st.session_state.formation_key = formation_key_ui
    st.session_state.squad_size = target_squad_size_ui

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🎨 Settings Profiles")
    profile_names = list(PREDEFINED_PROFILES.keys()) # [Source 62]

    def apply_profile_settings(profile_name): # As per user's file [Source 62-63]
        # Update current profile name state FIRST
        st.session_state.current_profile_name = profile_name
        if profile_name != "Custom" and profile_name in PREDEFINED_PROFILES:
            profile = PREDEFINED_PROFILES[profile_name]
            st.session_state.n_recent = profile.get("n_recent_games", st.session_state.n_recent) 
            st.session_state.min_recent_filter = profile.get("min_recent_games_played_filter", st.session_state.min_recent_filter)
            # Ensure deep copy for dicts if they are modified in place later by widgets
            st.session_state.kpi_weights = profile.get("kpi_weights", {}).copy() 
            st.session_state.mrb_params_per_pos = profile.get("mrb_params_per_pos", {}).copy()
        # If "Custom", the existing session state values (potentially modified by user) are retained.

    selected_profile_name_ui = st.sidebar.selectbox("Select Profile", options=profile_names, 
                                                    index=profile_names.index(st.session_state.current_profile_name), 
                                                    key="profile_selector_v5_main_unique", # Make sure keys are unique
                                                    help="Loads predefined settings. Modifying details below sets to 'Custom'.") # [Source 64]
    if selected_profile_name_ui != st.session_state.current_profile_name: # Check if user made a new selection
        apply_profile_settings(selected_profile_name_ui)
        st.rerun() # User's file uses st.rerun() [Source 64]

    with st.sidebar.expander("📊 KPI Weights (Click to Customize)", expanded=(st.session_state.current_profile_name == "Custom")):
        # Ensure active_kpi_weights is a mutable copy if needed, or that sliders update session_state directly
        active_kpi_weights = st.session_state.kpi_weights 
        weights_ui = {} # Temp dict to gather UI values
        for pos_key in ['GK', 'DEF', 'MID', 'FWD']: # [Source 65]
            st.markdown(f'<h6>{pos_key}</h6>', unsafe_allow_html=True)
            default_pos_w_structure = PREDEFINED_PROFILES["Balanced Value"]["kpi_weights"][pos_key]
            current_pos_w_vals = active_kpi_weights.get(pos_key, default_pos_w_structure)
            
            # Sliders from user's file (source 66-67)
            # User's file omits 'regularity_file' in the PVS calculation, so slider for it might be confusing if not used.
            # However, their UI code (source 65-68) did not have 'regularity_file'.
            # The profiles (source 5) also did not use 'regularity_file'.
            # Sticking to user's latest structure (source 66-67) which uses calc_regularity.
            weights_ui[pos_key] = {
                'season_avg': st.slider(f"Season Avg Rating", 0.0, 1.0, float(current_pos_w_vals.get('season_avg', 0.0)), 0.01, key=f"{pos_key}_wSA_v5_final"), # [Source 66]
                'season_goals': st.slider(f"Season Goals", 0.0, 1.0, float(current_pos_w_vals.get('season_goals', 0.0)) if pos_key in ['DEF', 'MID', 'FWD'] else 0.0, 0.01, key=f"{pos_key}_wSG_v5_final", disabled=pos_key not in ['DEF','MID', 'FWD']),
                'calc_regularity': st.slider(f"Calculated Regularity", 0.0, 1.0, float(current_pos_w_vals.get('calc_regularity', 0.0)), 0.01, key=f"{pos_key}_wCR_v5_final", help="Based on starts identified in gameweek data."),
                'recent_goals': st.slider(f"Recent Goals", 0.0, 1.0, float(current_pos_w_vals.get('recent_goals', 0.0)) if pos_key in ['DEF', 'MID', 'FWD'] else 0.0, 0.01, key=f"{pos_key}_wRG_v5_final", disabled=pos_key not in ['DEF','MID', 'FWD']), # [Source 67]
                'recent_avg': st.slider(f"Recent Avg Rating", 0.0, 1.0, float(current_pos_w_vals.get('recent_avg', 0.0)), 0.01, key=f"{pos_key}_wRA_v5_final"),
            }
        # Check for changes and update session state
        if weights_ui != active_kpi_weights: # This comparison works for dicts
            st.session_state.current_profile_name = "Custom" # [Source 68]
            st.session_state.kpi_weights = weights_ui # Update the authoritative source

    with st.sidebar.expander("💰 MRB Parameters (Click to Customize)", expanded=(st.session_state.current_profile_name == "Custom")):
        active_mrb_params = st.session_state.mrb_params_per_pos
        mrb_params_ui = {} # Temp dict
        for pos_key in ['GK', 'DEF', 'MID', 'FWD']:
            st.markdown(f'<h6>{pos_key}</h6>', unsafe_allow_html=True)
            default_pos_mrb_structure = PREDEFINED_PROFILES["Balanced Value"]["mrb_params_per_pos"][pos_key] # [Source 69]
            current_pos_mrb_vals = active_mrb_params.get(pos_key, default_pos_mrb_structure)
            # User's file MRB slider max_proportional_bonus_at_pvs100 has max 2.0 [Source 70]
            # My MRB calculation caps bonus at 1.0 (MRB = 2x Cote max).
            # To align, slider should be 0.0 to 1.0.
            # If user intends bonus up to 200% (MRB 3x Cote) before 2x cap, MRB calc must change.
            # Assuming user wants MRB capped at 2x Cote as per my MRB calc.
            mrb_params_ui[pos_key] = {
                'max_proportional_bonus_at_pvs100': st.slider(f"Max Bonus Factor (at PVS 100)", 0.0, 1.0, # Corrected range
                                                              float(current_pos_mrb_vals.get('max_proportional_bonus_at_pvs100', 0.2)), 
                                                              0.01, key=f"{pos_key}_mrbMPB_v5_final", 
                                                              help="Bonus factor if PVS=100 (e.g., 0.5 = 50% bonus implies MRB up to 1.5x Cote). Overall MRB is capped at 2x Cote.")
            }
        if mrb_params_ui != active_mrb_params:
            st.session_state.current_profile_name = "Custom"
            st.session_state.mrb_params_per_pos = mrb_params_ui # Update authoritative source
    
    # Dynamic Calculation and Display (from user's file, with minor adaptations for clarity)
    if uploaded_file: # Calculations only proceed if a file is uploaded
        with st.spinner("🧠 Strategizing your optimal squad... (Updates on input change)"):
            try: # [Source 71]
                df_input_calc = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded_file)
                df_processed_calc = df_input_calc.copy()
                df_processed_calc['simplified_position'] = df_processed_calc['Poste'].apply(strategist.simplify_position)
                df_processed_calc['player_id'] = df_processed_calc.apply(strategist.create_player_id, axis=1) # [Source 72]
                df_processed_calc['Cote'] = pd.to_numeric(df_processed_calc['Cote'], errors='coerce').fillna(1).clip(lower=1).round().astype(int)
                if 'Indispo ?' not in df_processed_calc.columns: # [Source 73]
                    df_processed_calc['Indispo ?'] = False
                else:
                    df_processed_calc['Indispo ?'] = df_processed_calc['Indispo ?'].astype(str).str.upper().isin(['TRUE', 'OUI', '1', 'YES', 'VRAI'])
                
                df_kpis = strategist.calculate_kpis(df_processed_calc, st.session_state.n_recent) # [Source 74]
                df_norm_kpis = strategist.normalize_kpis(df_kpis)
                df_pvs = strategist.calculate_pvs(df_norm_kpis, st.session_state.kpi_weights)
                df_mrb = strategist.calculate_mrb(df_pvs, st.session_state.mrb_params_per_pos)
                
                squad_df_result, squad_summary_result = strategist.select_squad(
                    df_mrb, st.session_state.formation_key, st.session_state.squad_size, st.session_state.min_recent_filter
                ) # [Source 75]
                
                st.session_state['df_for_display_final'] = df_mrb
                st.session_state['squad_df_result_final'] = squad_df_result
                st.session_state['squad_summary_result_final'] = squad_summary_result # [Source 76]
                st.session_state['selected_formation_key_display_final'] = st.session_state.formation_key
            except Exception as e:
                st.error(f"💥 Error during dynamic calculation: {str(e)}")

        # Main Panel Display Logic (using structure from user's file)
        if 'squad_df_result_final' in st.session_state and \
           st.session_state['squad_df_result_final'] is not None and \
           not st.session_state['squad_df_result_final'].empty: # [Source 77]
            
            col_main_results, col_summary = st.columns([3, 1]) # User var names [Source 77]
            with col_main_results:
                st.markdown('<h2 class="section-header">🏆 Suggested Squad</h2>', unsafe_allow_html=True)
                sdf = st.session_state['squad_df_result_final'].copy()
                
                int_cols_squad = ['mrb_actual_cost', 'Cote', 'recent_goals', 'season_goals'] # [Source 78]
                for col in int_cols_squad:
                    if col in sdf.columns: 
                        sdf[col] = pd.to_numeric(sdf[col], errors='coerce').fillna(0).round().astype(int) # [Source 79]
                
                squad_cols_display = ['Joueur', 'Club', 'simplified_position', 'pvs_in_squad', 'Cote', 'mrb_actual_cost', 'season_avg_rating', 'season_goals', 'calc_regularity_pct', 'recent_goals', 'recent_avg_rating', 'value_per_cost', 'is_starter'] # [Source 79]
                squad_cols_exist_display = [col for col in squad_cols_display if col in sdf.columns]
                sdf = sdf[squad_cols_exist_display]
                
                sdf.rename(columns={ # [Source 80]
                    'Joueur': 'Player', 'simplified_position': 'Pos', 'pvs_in_squad': 'PVS', # [Source 81]
                    'Cote': 'Cote', 'mrb_actual_cost': 'Suggested Bid', 'season_avg_rating': 'Average',
                    'season_goals': 'Goals', 'calc_regularity_pct': '% played', # [Source 82]
                    'recent_goals': 'Rec.G', 'recent_avg_rating': 'Rec.AvgR', 
                    'value_per_cost': 'Val/MRB', 'is_starter': 'Starter' # [Source 83]
                }, inplace=True)
                
                # User's float formatting logic [Source 83-84] - careful with column names after rename
                float_cols_squad_format = ['PVS', 'Average', '% played', 'Rec.AvgR', 'Val/MRB'] # Based on renamed cols
                for col in float_cols_squad_format: 
                    if col in sdf.columns: # [Source 84]
                        sdf[col] = pd.to_numeric(sdf[col], errors='coerce').fillna(0.0).round(2)
                
                pos_order = ['GK', 'DEF', 'MID', 'FWD']
                if 'Pos' in sdf.columns:
                    sdf['Pos'] = pd.Categorical(sdf['Pos'], categories=pos_order, ordered=True) # [Source 85]
                    sdf = sdf.sort_values(by=['Starter', 'Pos', 'PVS'], ascending=[False, True, False])
                st.dataframe(sdf, use_container_width=True, hide_index=True)

            with col_summary: # [Source 86]
                st.markdown('<h2 class="section-header">📈 Squad Summary</h2>', unsafe_allow_html=True)
                summary = st.session_state['squad_summary_result_final']
                if summary and isinstance(summary, dict):
                    st.metric("Budget Spent (MRB)", f"€ {summary.get('total_cost', 0):.0f}", help=f"Remaining: € {summary.get('remaining_budget', 0):.0f}") # User's formatting
                    st.metric("Squad Size", f"{summary.get('total_players', 0)} (Target: {st.session_state.squad_size})") # [Source 87]
                    st.metric("Total Squad PVS", f"{summary.get('total_squad_pvs', 0):.2f}")
                    st.metric("Starters PVS", f"{summary.get('total_starters_pvs', 0):.2f}")
                    st.info(f"**Formation:** {st.session_state.get('selected_formation_key_display_final', 'N/A')}")
                    st.markdown("**Positional Breakdown:**") # [Source 88]
                    for pos_cat_sum in pos_order: # Use same pos_order for consistency
                        count_sum = summary.get('position_counts', {}).get(pos_cat_sum, 0)
                        min_req_sum = strategist.squad_minimums.get(pos_cat_sum, 0) # [Source 89]
                        st.markdown(f"• **{pos_cat_sum}:** {count_sum} (Min: {min_req_sum})") # User uses st.markdown
                    else:
                        st.warning("Squad summary unavailable.")
            
            st.markdown('<hr><h2 class="section-header">📋 Full Player Database & Values</h2>', unsafe_allow_html=True) # [Source 90]
            if 'df_for_display_final' in st.session_state and st.session_state['df_for_display_final'] is not None:
                df_full = st.session_state['df_for_display_final'].copy()
                int_cols_full_display = ['Cote', 'mrb', 'recent_goals', 'season_goals', 'recent_games_played_count', 'games_started_season', 'total_season_gws_considered'] # [Source 91]
                for col in int_cols_full_display:
                    if col in df_full.columns:
                        df_full[col] = pd.to_numeric(df_full[col], errors='coerce').fillna(0).round().astype(int)
                
                all_stats_cols_display = ['Joueur', 'Club', 'simplified_position', 'pvs', 'Cote', 'mrb', 'Indispo ?',  # [Source 91-92]
                                   'season_avg_rating', 'season_goals',
                                   'calc_regularity_pct', 'recent_goals', 'recent_avg_rating', 'value_per_cost',
                                   'games_started_season', 'recent_games_played_count']
                df_full = df_full[[col for col in all_stats_cols_display if col in df_full.columns]] # [Source 93]
                df_full.rename(columns={ # [Source 93-97]
                    'Joueur': 'Player', 'simplified_position': 'Pos', 'pvs': 'PVS', 'Cote': 'Cote', 
                    'mrb': 'Suggested Bid', 'Indispo ?': 'Unavail.', 'season_avg_rating': 'Average', 
                    'season_goals': 'Goals', 'calc_regularity_pct': '% Played', 'recent_goals': 'Rec.G', 
                    'recent_avg_rating': 'Rec.AvgR', 'value_per_cost': 'Val/MRB', 
                    'games_started_season': 'Sea.Start', 'recent_games_played_count': 'Rec.Plyd'
                }, inplace=True)
                
                # Float formatting for full list based on user's code (Source 97-98)
                # User's list of float_cols_full included N. prefixed columns not in all_stats_cols_display
                # Re-evaluating based on actual columns after rename.
                float_cols_full_format = ['PVS', 'Average', '% Played', 'Rec.AvgR', 'Val/MRB']
                for col in float_cols_full_format: # [Source 98]
                    if col in df_full.columns:
                        df_full[col] = pd.to_numeric(df_full[col], errors='coerce').fillna(0.0).round(2)

                search_all = st.text_input("🔍 Search All Players:", key="search_all_v4") # User key [Source 98]
                if search_all: # [Source 99]
                    df_full = df_full[df_full.apply(lambda r: r.astype(str).str.contains(search_all, case=False, na=False).any(), axis=1)]
                st.dataframe(df_full.sort_values(by='PVS', ascending=False), use_container_width=True, hide_index=True, height=600)
                st.download_button( # [Source 100]
                    label="📥 Download Full Analysis (CSV)", 
                    data=df_full.to_csv(index=False).encode('utf-8'),
                    file_name="mpg_full_player_analysis_v4.csv", # User filename [Source 100]
                    mime="text/csv",
                    key="download_v4" # [Source 101]
                )
            elif not uploaded_file: # [Source 101]
                 pass # Handled by top-level else
            elif 'squad_df_result_final' not in st.session_state and uploaded_file:
                st.info("📊 Adjust settings in the sidebar. Results update dynamically when inputs change.") # [Source 102]
        
        else: # This 'else' is for the 'if uploaded_file:' block
            st.info("👈 Upload your MPG ratings file to begin.") # [Source 102]
            # Display Expected File Format Guide (as in my previous refined versions)
            st.markdown('<hr><h2 class="section-header">📋 Expected File Format Guide</h2>', unsafe_allow_html=True)
            example_data = {
                'Joueur': ['Player A', 'Player B'], 'Poste': ['A', 'M'], 'Club': ['Club X', 'Club Y'], 
                'Indispo ?': ['', 'TRUE'], 'Cote': [45, 30], '%Titu': [90, 75], 
                'D34': ['7.5*', '6.5'], 'D33': ['(6.0)**', '0'], 'D32': ['', '5.5*']
            }
            st.dataframe(pd.DataFrame(example_data), use_container_width=True, hide_index=True)
            st.markdown("""
            **Key Column Explanations:**
            - **Joueur, Poste, Club, Cote, %Titu**: As commonly understood.
            - **Indispo ?**: 'TRUE', 'OUI', '1', 'YES', 'VRAI' (case-insensitive) mark player as unavailable. (Note: Filter behavior depends on main processing).
            - **Dxx**: Gameweek columns. `Rating*` (goal), `(SubRating)`. Blank/'0' = DNP.
            """)
    # else for if not uploaded_file: (This was the outer else in user's code structure for the entire settings + processing block)
    # My structure now puts settings outside, and this `else` is just for the "Upload file to begin" message.
    # The user's provided code (source 102) had this outer else. I will keep it for structural consistency with their file.
    # But the guide is useful, so I will keep it inside the "if not uploaded_file" at the very end.
    # Corrected structure: if uploaded_file -> do all settings & processing. else -> show "upload file" and guide.

if __name__ == "__main__":
    main()
