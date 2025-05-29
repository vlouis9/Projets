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

    def select_squad(self, df: pd.DataFrame, formation_key: str, target_squad_size: int,
                     min_recent_games_played: int) -> Tuple[pd.DataFrame, Dict]: # [Source 33]
        """
        Selects a squad by first prioritizing player counts and PVS (potentially going over budget),
        then iteratively adjusts to meet the budget, and finally tries to maximize PVS within budget.
        """
        # --- Initial Filtering ---
        eligible_df_initial = df.copy()
        if min_recent_games_played > 0:
            eligible_df_initial = eligible_df_initial[eligible_df_initial['recent_games_played_count'] >= min_recent_games_played]
        
        # User mentioned removing 'Indispo ?' filter in conversation, 
        # but their provided code (mercato_mpg_gemini.txt, line 33-34) does not explicitly filter it out in select_squad.
        # I will retain the structure from their file, which doesn't filter 'Indispo ?' here.
        # If 'Indispo ?' should be filtered, it was handled in main() before calling processing.
        # The provided code source 33-34 for select_squad has:
        # if 'Indispo ?' in eligible_df_initial.columns: # This line is commented out in user's file
        # eligible_df_initial = eligible_df_initial[~eligible_df_initial['Indispo ?']] # This line is commented out
        # So, I will follow the user's file which means no 'Indispo ?' filtering inside select_squad.

        if eligible_df_initial.empty: # [Source 34]
            return pd.DataFrame(), {}
        
        eligible_df = eligible_df_initial.drop_duplicates(subset=['player_id']).copy()
        eligible_df['mrb'] = eligible_df['mrb'].astype(int)

        # --- Data structure for currently selected players ---
        # List of dicts: {'player_id': str, 'mrb': int, 'pvs': float, 'pos': str, 'is_starter': bool}
        current_squad_list_of_dicts = []
        
        # --- Helper to manage additions and removals, ensuring no duplicates ---
        def add_player_to_current_squad(player_row_data, is_starter_role):
            player_id_to_add = player_row_data['player_id']
            if player_id_to_add in [p['player_id'] for p in current_squad_list_of_dicts]:
                return False # Already in squad
            
            current_squad_list_of_dicts.append({
                'player_id': player_id_to_add,
                'mrb': int(player_row_data['mrb']),
                'pvs': float(player_row_data['pvs']),
                'pos': player_row_data['simplified_position'],
                'is_starter': is_starter_role
            })
            return True

        def remove_player_from_current_squad(player_id_to_remove):
            nonlocal current_squad_list_of_dicts
            initial_len = len(current_squad_list_of_dicts)
            current_squad_list_of_dicts = [p for p in current_squad_list_of_dicts if p['player_id'] != player_id_to_remove]
            return len(current_squad_list_of_dicts) < initial_len
            
        def get_current_squad_player_ids():
            return {p['player_id'] for p in current_squad_list_of_dicts}

        def get_current_pos_counts():
            counts = {pos_key: 0 for pos_key in self.squad_minimums.keys()}
            for p_dict in current_squad_list_of_dicts:
                counts[p_dict['pos']] = counts.get(p_dict['pos'], 0) + 1
            return counts
            
        # --- Phase A: Initial High-PVS Squad Construction (Potentially Over Budget) ---
        all_players_sorted_pvs = eligible_df.sort_values(by='pvs', ascending=False)

        # A1: Select Starters
        starters_map = self.formations[formation_key].copy()
        for _, player_row in all_players_sorted_pvs.iterrows():
            pos = player_row['simplified_position']
            if player_row['player_id'] not in get_current_squad_player_ids() and starters_map.get(pos, 0) > 0:
                if add_player_to_current_squad(player_row, True):
                    starters_map[pos] -= 1
        
        # A2: Fulfill Overall Squad Positional Minimums
        current_counts_ph_a2 = get_current_pos_counts()
        for pos, min_needed in self.squad_minimums.items():
            while current_counts_ph_a2.get(pos, 0) < min_needed:
                candidate = all_players_sorted_pvs[
                    (all_players_sorted_pvs['simplified_position'] == pos) &
                    (~all_players_sorted_pvs['player_id'].isin(get_current_squad_player_ids()))
                ].head(1)
                if candidate.empty: break 
                if add_player_to_current_squad(candidate.iloc[0], False): # Added as bench
                    current_counts_ph_a2 = get_current_pos_counts() # Recalculate counts
                else: break # Should not happen if isin check is robust

        # A3: Complete to Target Squad Size
        while len(current_squad_list_of_dicts) < target_squad_size:
            candidate = all_players_sorted_pvs[
                ~all_players_sorted_pvs['player_id'].isin(get_current_squad_player_ids())
            ].head(1)
            if candidate.empty: break
            add_player_to_current_squad(candidate.iloc[0], False) # Added as bench

        # Calculate initial state
        current_total_mrb = sum(p['mrb'] for p in current_squad_list_of_dicts)
        
        # --- Phase B: Iterative Budget Conformance (If Over Budget) ---
        # This phase is complex to implement perfectly to "minimize PVS loss"
        # For this version, we'll try a simpler downgrade: swap most expensive non-starters
        # with cheaper alternatives if significantly over budget.

        max_budget_iterations = 3 * target_squad_size # Limit iterations
        iteration_count = 0
        
        while current_total_mrb > self.budget and iteration_count < max_budget_iterations:
            iteration_count += 1
            made_a_change_in_pass = False
            
            # Sort current squad: non-starters first, then by MRB descending (to target expensive bench)
            # For simplicity, just iterate through non-starters sorted by MRB desc.
            # The 'is_starter' field in current_squad_list_of_dicts reflects the initial assignment.
            
            squad_for_downgrade_candidates = sorted(
                [p for p in current_squad_list_of_dicts if not p['is_starter']], 
                key=lambda x: x['mrb'], 
                reverse=True
            )
            
            if not squad_for_downgrade_candidates and current_total_mrb > self.budget:
                # If only starters are left and still over budget, pick any starter (not ideal)
                 squad_for_downgrade_candidates = sorted(current_squad_list_of_dicts, key=lambda x: x['mrb'], reverse=True)


            best_downgrade_this_pass = None # (old_pid, new_pid, mrb_saved, pvs_diff, new_player_row_data)
                                            # pvs_diff = new_pvs - old_pvs (want to minimize negative impact)

            for old_player_dict in squad_for_downgrade_candidates:
                old_player_id = old_player_dict['player_id']
                old_player_pos = old_player_dict['pos']
                old_player_mrb = old_player_dict['mrb']
                old_player_pvs = old_player_dict['pvs']

                # Find cheaper replacements NOT in squad
                potential_replacements = eligible_df[
                    (eligible_df['simplified_position'] == old_player_pos) &
                    (~eligible_df['player_id'].isin(get_current_squad_player_ids())) &
                    (eligible_df['mrb'] < old_player_mrb) # Must be cheaper
                ].sort_values(by='pvs', ascending=False) # Get best PVS among cheaper

                if not potential_replacements.empty:
                    for _, new_player_candidate_row in potential_replacements.iterrows():
                        mrb_saved_by_swap = old_player_mrb - new_player_candidate_row['mrb']
                        pvs_difference_by_swap = new_player_candidate_row['pvs'] - old_player_pvs
                        
                        # Heuristic: Maximize MRB saved, minimize PVS loss (or maximize PVS gain if possible)
                        # Score = mrb_saved - (pvs_loss * weight) -> Higher is better
                        # For simplicity: pick the one that saves most MRB with highest PVS
                        current_swap_score = mrb_saved_by_swap + (pvs_difference_by_swap * 0.1) # Small weight for pvs diff

                        if best_downgrade_this_pass is None or current_swap_score > (best_downgrade_this_pass[2] + (best_downgrade_this_pass[3]*0.1)):
                            best_downgrade_this_pass = (
                                old_player_id, 
                                new_player_candidate_row['player_id'], 
                                mrb_saved_by_swap, 
                                pvs_difference_by_swap,
                                new_player_candidate_row # Store the full row for adding
                            )
                        break # Take the first (best PVS) cheaper replacement for this iteration
            
            if best_downgrade_this_pass:
                old_pid, new_pid, mrb_saved, pvs_diff, new_player_data_row = best_downgrade_this_pass
                
                # Find original starter status of old_pid to preserve it for the new player if it was a starter downgrade
                old_player_original_starter_status = False
                for p_dict_old in current_squad_list_of_dicts:
                    if p_dict_old['player_id'] == old_pid:
                        old_player_original_starter_status = p_dict_old['is_starter']
                        break

                if remove_player_from_current_squad(old_pid):
                    if add_player_to_current_squad(new_player_data_row, old_player_original_starter_status): # Add new player
                        current_total_mrb -= mrb_saved
                        st.caption(f"Budget Downgrade: Swapped. Saved €{mrb_saved}. PVS change: {pvs_diff:.2f}. New Total MRB: {current_total_mrb}")
                        made_a_change_in_pass = True
                    else: # Should not happen if logic is correct
                        st.warning("Failed to add replacement during downgrade.")
                        break 
                else: # Should not happen
                    st.warning(f"Failed to remove {old_pid} during downgrade.")
                    break
            
            if not made_a_change_in_pass:
                if current_total_mrb > self.budget:
                     st.warning(f"Could not get under budget after {iteration_count} iterations. Final MRB: {current_total_mrb}. Budget is {self.budget}")
                break # No beneficial downgrades found in this pass
        
        # --- Phase C: Final PVS Upgrade (Spend Remaining Budget) ---
        # Similar to user's code (Source 1, lines 43-53), adapted for current_squad_list_of_dicts
        budget_remaining_for_upgrades = self.budget - current_total_mrb
        if budget_remaining_for_upgrades > 5 and len(current_squad_list_of_dicts) == target_squad_size:
            max_upgrade_passes = 5 
            upgrade_pass_count = 0
            made_an_upgrade_in_pass = True

            while made_an_upgrade_in_pass and upgrade_pass_count < max_upgrade_passes and budget_remaining_for_upgrades > 5:
                made_an_upgrade_in_pass = False
                upgrade_pass_count += 1
                
                # Try to upgrade non-starters first
                squad_for_upgrade_candidates = sorted(
                    [p for p in current_squad_list_of_dicts if not p['is_starter']], 
                    key=lambda x: x['pvs'] # Lowest PVS non-starters first
                )
                if not squad_for_upgrade_candidates: # If all are starters, try upgrading any
                    squad_for_upgrade_candidates = sorted(current_squad_list_of_dicts, key=lambda x:x['pvs'])


                for old_player_dict_upg in squad_for_upgrade_candidates:
                    if budget_remaining_for_upgrades <= 5: break

                    best_upgrade_candidate_row = None
                    max_pvs_gain_for_budget = -1

                    potential_upgrades_pool = eligible_df[
                        (eligible_df['simplified_position'] == old_player_dict_upg['pos']) &
                        (~eligible_df['player_id'].isin(get_current_squad_player_ids())) &
                        (eligible_df['pvs'] > old_player_dict_upg['pvs']) # Must be PVS improvement
                    ].sort_values(by='pvs', ascending=False)

                    for _, new_player_candidate_row_upg in potential_upgrades_pool.iterrows():
                        cost_of_this_upgrade = new_player_candidate_row_upg['mrb'] - old_player_dict_upg['mrb']
                        if cost_of_this_upgrade <= budget_remaining_for_upgrades and cost_of_this_upgrade >= 0 : # Affordable & not cheaper (unless PVS gain is huge)
                            pvs_gain_this_swap = new_player_candidate_row_upg['pvs'] - old_player_dict_upg['pvs']
                            # Simple: pick the one that gives most PVS gain and is affordable
                            if pvs_gain_this_swap > max_pvs_gain_for_budget:
                                max_pvs_gain_for_budget = pvs_gain_this_swap
                                best_upgrade_candidate_row = new_player_candidate_row_upg
                    
                    if best_upgrade_candidate_row is not None:
                        cost_of_best_upgrade = best_upgrade_candidate_row['mrb'] - old_player_dict_upg['mrb']
                        
                        # Preserve starter status of the slot being upgraded
                        original_starter_status_upg = old_player_dict_upg['is_starter']

                        if remove_player_from_current_squad(old_player_dict_upg['player_id']):
                            if add_player_to_current_squad(best_upgrade_candidate_row, original_starter_status_upg):
                                current_total_mrb += cost_of_best_upgrade
                                budget_remaining_for_upgrades = self.budget - current_total_mrb
                                st.caption(f"Budget Upgrade: Swapped. Cost increase €{cost_of_best_upgrade}. PVS gain: {max_pvs_gain_for_budget:.2f}. New Total MRB: {current_total_mrb}")
                                made_an_upgrade_in_pass = True
                                break # Restart scan for upgrades on the modified squad
                            else: st.warning("Failed to add replacement during upgrade.")
                        else: st.warning("Failed to remove old player during upgrade.")
                if not made_an_upgrade_in_pass : break


        # --- Final Squad Construction from current_squad_list_of_dicts ---
        if not current_squad_list_of_dicts: return pd.DataFrame(), {}
        
        final_squad_player_ids = get_current_squad_player_ids()
        # Get full data for selected players from the original eligible_df (which has all columns)
        final_squad_df_base = eligible_df[eligible_df['player_id'].isin(final_squad_player_ids)].copy()

        # Create a DataFrame from current_squad_list_of_dicts to easily map mrb_cost and is_starter
        details_df = pd.DataFrame(current_squad_list_of_dicts)
        details_df.rename(columns={'mrb': 'mrb_actual_cost', 'pvs':'pvs_in_squad', 'pos':'final_pos', 'is_starter':'is_starter_final'}, inplace=True)
        
        final_squad_df = pd.merge(final_squad_df_base, 
                                  details_df[['player_id', 'mrb_actual_cost', 'pvs_in_squad', 'is_starter_final', 'final_pos']], 
                                  on='player_id', 
                                  how='left')
        
        # Ensure correct starter status after all swaps (re-evaluate based on PVS for formation)
        # This step ensures the best PVS players for the formation are marked as starters.
        final_starter_ids_set_final_pass = set()
        temp_formation_needs_final = self.formations[formation_key].copy()
        # Sort final squad by PVS to pick starters
        final_squad_df_sorted_for_starters = final_squad_df.sort_values(by='pvs_in_squad', ascending=False)

        for _, player_row_final in final_squad_df_sorted_for_starters.iterrows():
            pos_final = player_row_final['simplified_position'] # Use original simplified_position for consistency
            player_id_final = player_row_final['player_id']
            if temp_formation_needs_final.get(pos_final, 0) > 0:
                if player_id_final not in final_starter_ids_set_final_pass : # ensure player is not already a starter for another role if positions overlap
                    final_starter_ids_set_final_pass.add(player_id_final)
                    temp_formation_needs_final[pos_final] -=1
        
        final_squad_df['is_starter'] = final_squad_df['player_id'].isin(final_starter_ids_set_final_pass)
        final_squad_df.drop(columns=['is_starter_final', 'final_pos'], errors='ignore', inplace=True)


        final_total_mrb = final_squad_df['mrb_actual_cost'].sum()
        summary = {
            'total_players': len(final_squad_df),
            'total_cost': int(final_total_mrb),
            'remaining_budget': int(self.budget - final_total_mrb),
            'position_counts': final_squad_df['simplified_position'].value_counts().to_dict(),
            'total_squad_pvs': round(final_squad_df['pvs_in_squad'].sum(), 2),
            'total_starters_pvs': round(final_squad_df[final_squad_df['is_starter']]['pvs_in_squad'].sum(), 2)
        }
        
        # Final check: if any positional minimum is not met, this is an issue
        final_pos_counts_check = summary['position_counts']
        for pos_check, min_val_check in self.squad_minimums.items():
            if final_pos_counts_check.get(pos_check,0) < min_val_check:
                st.error(f"CRITICAL ERROR in squad selection: Position {pos_check} minimum not met! ({final_pos_counts_check.get(pos_check,0)}/{min_val_check})")

        return final_squad_df, summary

# ... (rest of the main() function from mercato_mpg_gemini.txt, from line 55 onward) ...
# Ensure the main() function calls the strategist methods correctly and handles UI.
# The UI part for KPI weights needs to map to 'regularity_file' if it's used in profiles/PVS calc.
# The user's PVS calculation (source 28) currently uses 'calc_regularity' but not 'regularity_file'.
# I will ensure the KPI weight sliders in main() match what calculate_pvs uses.
# The user's `mercato_mpg_gemini.txt`'s `calculate_pvs` (source 27-29) uses `w.get('calc_regularity', 0)`
# but does NOT use `w.get('regularity_file', 0)`.
# The profiles in their file (source 5-12) also omit `regularity_file` from kpi_weights.
# So, the slider for `regularity_file` in the UI should be removed or adapted if it's not used in PVS.
# For now, I will keep the PVS calculation as per user's file structure.
# The UI sliders for KPI weights in the provided user file (source 65-68) include sliders for both regularities.
# I will keep the UI sliders as they are in the user's file, but note that `regularity_file` weight won't affect PVS
# unless `calculate_pvs` is also changed to use it. My focus is only `select_squad`.

# (The main() function from line 55 of mercato_mpg_gemini.txt should be appended here)
# For this response, I will show the main function part that has critical changes regarding session state and apply_profile
# and the KPI weights section to ensure `calc_regularity` is there.

def main(): # [Source 55]
    st.markdown('<h1 class="main-header">🚀 MPG Auction Strategist v5 (Budget Focus)</h1>', unsafe_allow_html=True) # Updated title
    strategist = MPGAuctionStrategist()

    # Initialize session state correctly
    if 'current_profile_name' not in st.session_state:
        st.session_state.current_profile_name = "Balanced Value" # Default profile
        # Initialize other dependent session state variables based on this default
        profile_values = PREDEFINED_PROFILES[st.session_state.current_profile_name]
        st.session_state.n_recent = profile_values.get("n_recent_games", DEFAULT_N_RECENT_GAMES)
        st.session_state.min_recent_filter = profile_values.get("min_recent_games_played_filter", DEFAULT_MIN_RECENT_GAMES_PLAYED)
        st.session_state.kpi_weights = profile_values.get("kpi_weights", PREDEFINED_PROFILES["Balanced Value"]["kpi_weights"]) # Ensure deep copy or structure
        st.session_state.mrb_params_per_pos = profile_values.get("mrb_params_per_pos", PREDEFINED_PROFILES["Balanced Value"]["mrb_params_per_pos"])
    
    if 'formation_key' not in st.session_state: st.session_state.formation_key = DEFAULT_FORMATION
    if 'squad_size' not in st.session_state: st.session_state.squad_size = DEFAULT_SQUAD_SIZE


    # Sidebar UI Elements (matches user's file structure mostly)
    st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFr1EWtMR2tHq1FwHnCHqg2uXv1JMLYQlRZw&s", width=100)
    st.sidebar.markdown('<h2 class="section-header" style="margin-top:0;">⚙️ Controls & Settings</h2>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("📁 Upload MPG Ratings File (CSV/Excel)", type=['csv', 'xlsx', 'xls'], help="Joueur, Poste, Club, Cote, %Titu, Indispo?, Gameweeks (D1..D34).")

    # This conditional logic for `uploaded_file` was moved down in user's file, I'll keep it high for clarity of inputs
    # The UI elements for parameters should ideally be available to view/set even before upload,
    # and then calculations run when file is present.

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🌎 Global Data & Form Parameters") # [Source 59]
    n_recent_ui = st.sidebar.number_input("Recent Games Window (N)", min_value=1, max_value=38, value=st.session_state.n_recent, help="For 'Recent Form' KPIs. Avg of games *played* in this window.") # [Source 60]
    min_recent_filter_ui = st.sidebar.number_input("Filter: Min Games Played in Recent N Weeks", min_value=0, max_value=n_recent_ui, value=st.session_state.min_recent_filter, help=f"Exclude players with < this in '{n_recent_ui}' recent weeks. 0 = no filter.")
    if n_recent_ui != st.session_state.n_recent or min_recent_filter_ui != st.session_state.min_recent_filter:
        st.session_state.current_profile_name = "Custom" # If global params change, it's a custom setup
    st.session_state.n_recent = n_recent_ui
    st.session_state.min_recent_filter = min_recent_filter_ui

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 👥 Squad Building Parameters") # [Source 61]
    formation_key_ui = st.sidebar.selectbox("Preferred Starting Formation", options=list(strategist.formations.keys()), index=list(strategist.formations.keys()).index(st.session_state.formation_key))
    target_squad_size_ui = st.sidebar.number_input("Target Total Squad Size", min_value=strategist.squad_minimums_sum_val, max_value=30, value=st.session_state.squad_size)
    if formation_key_ui != st.session_state.formation_key or target_squad_size_ui != st.session_state.squad_size:
        st.session_state.current_profile_name = "Custom" # If squad params change, it's custom
    st.session_state.formation_key = formation_key_ui
    st.session_state.squad_size = target_squad_size_ui

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🎨 Settings Profiles")
    profile_names = list(PREDEFINED_PROFILES.keys()) # [Source 62]

    # apply_profile_settings function needs to be defined here or globally if not already.
    # Using the one from user's file, ensuring it's defined before use.
    def apply_profile_settings(profile_name): # As per user's file [Source 62-63]
        if profile_name != "Custom" and profile_name in PREDEFINED_PROFILES:
            profile = PREDEFINED_PROFILES[profile_name]
            # These global ones might be overridden by profiles
            st.session_state.n_recent = profile.get("n_recent_games", st.session_state.n_recent) 
            st.session_state.min_recent_filter = profile.get("min_recent_games_played_filter", st.session_state.min_recent_filter)
            st.session_state.kpi_weights = profile.get("kpi_weights", st.session_state.kpi_weights)
            st.session_state.mrb_params_per_pos = profile.get("mrb_params_per_pos", st.session_state.mrb_params_per_pos)
        st.session_state.current_profile_name = profile_name


    selected_profile_name_ui = st.sidebar.selectbox("Select Profile", options=profile_names, 
                                                    index=profile_names.index(st.session_state.current_profile_name), 
                                                    key="profile_selector_v5_main", # Unique key
                                                    help="Loads predefined settings. Modifying details below sets to 'Custom'.") # [Source 64]
    if selected_profile_name_ui != st.session_state.current_profile_name:
        apply_profile_settings(selected_profile_name_ui)
        st.rerun() # User's file uses st.rerun() [Source 64]

    with st.sidebar.expander("📊 KPI Weights (Click to Customize)", expanded=(st.session_state.current_profile_name == "Custom")):
        active_kpi_weights = st.session_state.kpi_weights
        weights_ui = {}
        for pos_key in ['GK', 'DEF', 'MID', 'FWD']: # [Source 65]
            st.markdown(f'<h6>{pos_key}</h6>', unsafe_allow_html=True)
            # Fallback to a default profile's structure if active_kpi_weights doesn't have the pos_key
            default_pos_w_structure = PREDEFINED_PROFILES["Balanced Value"]["kpi_weights"][pos_key]
            current_pos_w_vals = active_kpi_weights.get(pos_key, default_pos_w_structure)
            
            # Match sliders to user's file (source 66-67), which omits 'regularity_file'
            weights_ui[pos_key] = {
                'season_avg': st.slider(f"Season Avg Rating", 0.0, 1.0, float(current_pos_w_vals.get('season_avg', 0.0)), 0.01, key=f"{pos_key}_wSA_v5_main"), # [Source 66]
                'season_goals': st.slider(f"Season Goals", 0.0, 1.0, float(current_pos_w_vals.get('season_goals', 0.0)) if pos_key in ['DEF', 'MID', 'FWD'] else 0.0, 0.01, key=f"{pos_key}_wSG_v5_main", disabled=pos_key not in ['DEF', 'MID', 'FWD']),
                'calc_regularity': st.slider(f"Calculated Regularity", 0.0, 1.0, float(current_pos_w_vals.get('calc_regularity', 0.0)), 0.01, key=f"{pos_key}_wCR_v5_main", help="Based on starts identified in gameweek data."),
                'recent_goals': st.slider(f"Recent Goals", 0.0, 1.0, float(current_pos_w_vals.get('recent_goals', 0.0)) if pos_key in ['DEF', 'MID', 'FWD'] else 0.0, 0.01, key=f"{pos_key}_wRG_v5_main", disabled=pos_key not in ['DEF', 'MID', 'FWD']), # [Source 67]
                'recent_avg': st.slider(f"Recent Avg Rating", 0.0, 1.0, float(current_pos_w_vals.get('recent_avg', 0.0)), 0.01, key=f"{pos_key}_wRA_v5_main"),
            }
        if weights_ui != active_kpi_weights:
            st.session_state.current_profile_name = "Custom" # [Source 68]
            st.session_state.kpi_weights = weights_ui

    with st.sidebar.expander("💰 MRB Parameters (Click to Customize)", expanded=(st.session_state.current_profile_name == "Custom")):
        active_mrb_params = st.session_state.mrb_params_per_pos
        mrb_params_ui = {}
        for pos_key in ['GK', 'DEF', 'MID', 'FWD']:
            st.markdown(f'<h6>{pos_key}</h6>', unsafe_allow_html=True)
            default_pos_mrb_structure = PREDEFINED_PROFILES["Balanced Value"]["mrb_params_per_pos"][pos_key] # [Source 69]
            current_pos_mrb_vals = active_mrb_params.get(pos_key, default_pos_mrb_structure)
            # User's MRB param slider max_proportional_bonus_at_pvs100 goes up to 2.0 [Source 69-70]
            # This means bonus can be 200% of Cote, so MRB can be Cote * 3.
            # However, calculate_mrb caps MRB at 2x Cote. The slider should ideally be 0.0 to 1.0
            # if max_prop_bonus is the *bonus factor* and the 2x Cote cap is firm.
            # I will keep the slider 0.0-1.0 as previous and ensure logic handles it.
            # If user's file has 2.0 and they want MRB up to 3x Cote (before 2x cap), then MRB calc changes.
            # Sticking to MRB max 2x Cote overall as per my MRB calc. Max bonus factor = 1.0 for that.
            mrb_params_ui[pos_key] = {
                'max_proportional_bonus_at_pvs100': st.slider(f"Max Bonus Factor (at PVS 100)", 0.0, 1.0, # Corrected range
                                                              float(current_pos_mrb_vals.get('max_proportional_bonus_at_pvs100', 0.2)), 
                                                              0.01, key=f"{pos_key}_mrbMPB_v5_main", 
                                                              help="Bonus factor if PVS=100 (e.g., 0.5 = 50% bonus). MRB capped at 2x Cote.")
            }
        if mrb_params_ui != active_mrb_params:
            st.session_state.current_profile_name = "Custom"
        st.session_state.mrb_params_per_pos = mrb_params_ui
    
    # Dynamic Calculation and Display
    if uploaded_file:
        with st.spinner("🧠 Strategizing your optimal squad... (Updates on input change)"):
            try: # [Source 71]
                # File reading and initial processing from user's code [Source 71-73]
                df_input_calc = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded_file)
                df_processed_calc = df_input_calc.copy()
                df_processed_calc['simplified_position'] = df_processed_calc['Poste'].apply(strategist.simplify_position)
                df_processed_calc['player_id'] = df_processed_calc.apply(strategist.create_player_id, axis=1) # [Source 72]
                df_processed_calc['Cote'] = pd.to_numeric(df_processed_calc['Cote'], errors='coerce').fillna(1).clip(lower=1).round().astype(int)
                if 'Indispo ?' not in df_processed_calc.columns: # [Source 73]
                    df_processed_calc['Indispo ?'] = False
                else:
                    df_processed_calc['Indispo ?'] = df_processed_calc['Indispo ?'].astype(str).str.upper().isin(['TRUE', 'OUI', '1', 'YES', 'VRAI'])
                
                # Core calculations using session_state parameters
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

        # Main Panel Display Logic
        if 'squad_df_result_final' in st.session_state and \
           st.session_state['squad_df_result_final'] is not None and \
           not st.session_state['squad_df_result_final'].empty: # [Source 77]
            
            col_main_results_v5, col_summary_v5 = st.columns([3, 1])
            with col_main_results_v5:
                st.markdown('<h2 class="section-header">🏆 Suggested Squad</h2>', unsafe_allow_html=True)
                sdf_v5 = st.session_state['squad_df_result_final'].copy()
                
                int_cols_squad_v5 = ['mrb_actual_cost', 'Cote', 'recent_goals', 'season_goals'] # [Source 78]
                for col in int_cols_squad_v5:
                    if col in sdf_v5.columns: 
                        sdf_v5[col] = pd.to_numeric(sdf_v5[col], errors='coerce').fillna(0).round().astype(int) # [Source 79]
                
                # Use user's desired columns for squad display [Source 79]
                squad_cols_v5 = ['Joueur', 'Club', 'simplified_position', 'pvs_in_squad', 'Cote', 'mrb_actual_cost', 'season_avg_rating', 'season_goals', 'calc_regularity_pct', 'recent_goals', 'recent_avg_rating', 'value_per_cost', 'is_starter']
                squad_cols_exist_v5 = [col for col in squad_cols_v5 if col in sdf_v5.columns]
                sdf_v5 = sdf_v5[squad_cols_exist_v5]
                
                sdf_v5.rename(columns={ # [Source 80]
                    'Joueur': 'Player', 'simplified_position': 'Pos', 'pvs_in_squad': 'PVS', # [Source 81]
                    'Cote': 'Cote', 'mrb_actual_cost': 'Suggested Bid', 'season_avg_rating': 'Average',
                    'season_goals': 'Goals', 'calc_regularity_pct': '% played', # [Source 82]
                    'recent_goals': 'Rec.G', 'recent_avg_rating': 'Rec.AvgR', 
                    'value_per_cost': 'Val/MRB', 'is_starter': 'Starter' # [Source 83]
                }, inplace=True)
                
                # User's code does not include 'Reg.% (File)' or 'Sea.AvgR' in squad display float formatting
                # It has 'PVS', 'Rec.AvgR', 'Val/MRB'. I will adapt.
                # The float_cols_squad from user's file seems to refer to columns that might not exist after rename or selection [Source 83-84]
                # I will re-evaluate which columns need float formatting based on the renamed ones.
                float_cols_to_format_squad = ['PVS', 'Average', '% played', 'Rec.AvgR', 'Val/MRB']
                for col in float_cols_to_format_squad: # [Source 84]
                    if col in sdf_v5.columns:
                        sdf_v5[col] = pd.to_numeric(sdf_v5[col], errors='coerce').fillna(0.0).round(2)
                
                pos_order_v5 = ['GK', 'DEF', 'MID', 'FWD']
                if 'Pos' in sdf_v5.columns:
                    sdf_v5['Pos'] = pd.Categorical(sdf_v5['Pos'], categories=pos_order_v5, ordered=True) # [Source 85]
                    sdf_v5 = sdf_v5.sort_values(by=['Starter', 'Pos', 'PVS'], ascending=[False, True, False])
                st.dataframe(sdf_v5, use_container_width=True, hide_index=True)

            with col_summary_v5: # [Source 86]
                st.markdown('<h2 class="section-header">📈 Squad Summary</h2>', unsafe_allow_html=True)
                summary_v5 = st.session_state['squad_summary_result_final']
                if summary_v5 and isinstance(summary_v5, dict):
                    st.metric("Budget Spent (MRB)", f"€ {summary_v5.get('total_cost',0):.0f}", help=f"Remaining: € {summary_v5.get('remaining_budget',0):.0f}") # User file has / {self.budget}
                    st.metric("Squad Size", f"{summary_v5.get('total_players',0)} (Target: {st.session_state.squad_size})") # [Source 87]
                    st.metric("Total Squad PVS", f"{summary_v5.get('total_squad_pvs',0):.2f}")
                    st.metric("Starters PVS", f"{summary_v5.get('total_starters_pvs',0):.2f}")
                    st.info(f"**Formation:** {st.session_state.get('selected_formation_key_display_final', 'N/A')}")
                    st.markdown("**Positional Breakdown:**") # [Source 88]
                    for pos_cat in pos_order_v5:
                        count = summary_v5.get('position_counts', {}).get(pos_cat, 0)
                        min_req = strategist.squad_minimums.get(pos_cat, 0) # [Source 89]
                        st.markdown(f"• **{pos_cat}:** {count} (Min: {min_req})") # User file uses st.markdown
                    else:
                        st.warning("Squad summary unavailable.")
            
            st.markdown('<hr><h2 class="section-header">📋 Full Player Database & Values</h2>', unsafe_allow_html=True) # [Source 90]
            if 'df_for_display_final' in st.session_state and st.session_state['df_for_display_final'] is not None:
                df_full_v5 = st.session_state['df_for_display_final'].copy()
                int_cols_full_v5 = ['Cote', 'mrb', 'recent_goals', 'season_goals', 'recent_games_played_count', 'games_started_season', 'total_season_gws_considered'] # [Source 91]
                for col in int_cols_full_v5:
                    if col in df_full_v5.columns:
                        df_full_v5[col] = pd.to_numeric(df_full_v5[col], errors='coerce').fillna(0).round().astype(int)
                
                # User's column list for full display [Source 91-92]
                all_stats_cols_v5 = ['Joueur', 'Club', 'simplified_position', 'pvs', 'Cote', 'mrb', 'Indispo ?', 
                                   'season_avg_rating', 'season_goals',
                                   'calc_regularity_pct', 'recent_goals', 'recent_avg_rating', 'value_per_cost',
                                   'games_started_season', 'recent_games_played_count']
                df_full_v5 = df_full_v5[[col for col in all_stats_cols_v5 if col in df_full_v5.columns]] # [Source 93]
                df_full_v5.rename(columns={ # [Source 93-97]
                    'Joueur': 'Player', 'simplified_position': 'Pos', 'pvs': 'PVS', 'Cote': 'Cote', 
                    'mrb': 'Suggested Bid', 'Indispo ?': 'Unavail.', 'season_avg_rating': 'Average', 
                    'season_goals': 'Goals', 'calc_regularity_pct': '% Played', 'recent_goals': 'Rec.G', 
                    'recent_avg_rating': 'Rec.AvgR', 'value_per_cost': 'Val/MRB', 
                    'games_started_season': 'Sea.Start', 'recent_games_played_count': 'Rec.Plyd'
                }, inplace=True)
                
                # User's float formatting for full list [Source 97-98]
                # This list seems to have N. prefixed columns not in the selected all_stats_cols_v5 above
                # I will format floats based on the actual columns present after rename.
                float_cols_to_format_full = ['PVS', 'Average', '% Played', 'Rec.AvgR', 'Val/MRB']
                for col in float_cols_to_format_full: # [Source 98]
                    if col in df_full_v5.columns:
                        df_full_v5[col] = pd.to_numeric(df_full_v5[col], errors='coerce').fillna(0.0).round(2)

                search_all_v5 = st.text_input("🔍 Search All Players:", key="search_all_v5") # User key search_all_v4 [Source 98]
                if search_all_v5: # [Source 99]
                    df_full_v5 = df_full_v5[df_full_v5.apply(lambda r: r.astype(str).str.contains(search_all_v5, case=False, na=False).any(), axis=1)]
                st.dataframe(df_full_v5.sort_values(by='PVS', ascending=False), use_container_width=True, hide_index=True, height=600)
                st.download_button( # [Source 100]
                    label="📥 Download Full Analysis (CSV)", 
                    data=df_full_v5.to_csv(index=False).encode('utf-8'),
                    file_name="mpg_full_player_analysis_v5.csv", # Updated filename
                    mime="text/csv",
                    key="download_v5" # [Source 101]
                )
            elif not uploaded_file: # [Source 101]
                pass # Handled by top-level else
            elif 'squad_df_result_final' not in st.session_state and uploaded_file:
                st.info("📊 Adjust settings in the sidebar. Results update dynamically when inputs change.") # [Source 102]
        else: # No file uploaded
            st.info("👈 Upload your MPG ratings file to begin.") # [Source 102]
            # Display Expected File Format Guide (User's file does not have it here, but it's good practice)

    else: # No file uploaded (This block is from user's file [Source 102], but I'll keep the guide for clarity)
        st.info("👈 Upload your MPG ratings file to begin.")
        st.markdown('<hr><h2 class="section-header">📋 Expected File Format Guide</h2>', unsafe_allow_html=True)
        example_data = {
            'Joueur': ['Player A', 'Player B'], 'Poste': ['A', 'M'], 'Club': ['Club X', 'Club Y'], 
            'Indispo ?': ['', 'TRUE'], 'Cote': [45, 30], '%Titu': [90, 75], 
            'D34': ['7.5*', '6.5'], 'D33': ['(6.0)**', '0'], 'D32': ['', '5.5*']
        }
        st.dataframe(pd.DataFrame(example_data), use_container_width=True, hide_index=True)
        st.markdown("""
        **Key Column Explanations:**
        - **Joueur**: Player's full name.
        - **Poste**: Original position (G, D, DL, DC, M, MD, MO, A).
        - **Club**: Player's club.
        - **Indispo ?**: Availability. 'TRUE', 'OUI', '1', 'YES', 'VRAI' (case-insensitive) mark player as unavailable. Blank or other values mean available. (Note: User's latest `select_squad` does not seem to apply this filter directly from inside the function).
        - **Cote**: MPG Price (numeric).
        - **%Titu**: Titularisation percentage from file (numeric, e.g., 75 for 75%).
        - **Dxx (e.g., D34...D1)**: Gameweek columns.
            - Format: Rating (e.g., `6.5`), `(SubRating)` (e.g., `(5.0)`), `Rating*` (1 goal), `Rating**` (2 goals).
            - Blank or '0' cell = Did Not Play (DNP).
        """)


if __name__ == "__main__":
    main()
