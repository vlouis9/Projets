import streamlit as st
import pandas as pd
import numpy as np
import re # Keep for potential future use, though less critical now
from typing import Dict, List, Tuple, Optional, Set 

# Page configuration
st.set_page_config(
    page_title="MPG New Season Strategist", 
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS ( 그대로 유지 )
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

# --- Subjective KPIs ---
# These are the keys we'll use for the new subjective scores and their weights
KPI_PERFORMANCE = "PerformanceEstimation"
KPI_POTENTIAL = "PotentialEstimation"
KPI_REGULARITY = "RegularityEstimation"
KPI_GOALS = "GoalsEstimation"

# Normalized versions will be prefixed with "norm_"
SUBJECTIVE_KPI_COLUMNS = [KPI_PERFORMANCE, KPI_POTENTIAL, KPI_REGULARITY, KPI_GOALS]

# Constants
DEFAULT_SQUAD_SIZE = 20
DEFAULT_FORMATION = "4-4-2"

# --- Updated Predefined Profiles ---
PREDEFINED_PROFILES = {
    "Custom": "custom",
    "Balanced Value": {
        "kpi_weights": { # Weights for NEW Subjective KPIs
            'GK':  {KPI_PERFORMANCE: 0.50, KPI_POTENTIAL: 0.10, KPI_REGULARITY: 0.40, KPI_GOALS: 0.00},
            'DEF': {KPI_PERFORMANCE: 0.40, KPI_POTENTIAL: 0.15, KPI_REGULARITY: 0.35, KPI_GOALS: 0.10},
            'MID': {KPI_PERFORMANCE: 0.35, KPI_POTENTIAL: 0.20, KPI_REGULARITY: 0.25, KPI_GOALS: 0.20},
            'FWD': {KPI_PERFORMANCE: 0.30, KPI_POTENTIAL: 0.20, KPI_REGULARITY: 0.20, KPI_GOALS: 0.30}
        },
        "mrb_params_per_pos": { # MRB params can remain similar
            'GK': {'max_proportional_bonus_at_pvs100': 0.3},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.4},
            'MID': {'max_proportional_bonus_at_pvs100': 0.6}, 
            'FWD': {'max_proportional_bonus_at_pvs100': 0.8}
        }
    },
    "Focus on High Potential": {
        "kpi_weights": {
            'GK':  {KPI_PERFORMANCE: 0.20, KPI_POTENTIAL: 0.50, KPI_REGULARITY: 0.30, KPI_GOALS: 0.00},
            'DEF': {KPI_PERFORMANCE: 0.20, KPI_POTENTIAL: 0.50, KPI_REGULARITY: 0.20, KPI_GOALS: 0.10},
            'MID': {KPI_PERFORMANCE: 0.15, KPI_POTENTIAL: 0.50, KPI_REGULARITY: 0.15, KPI_GOALS: 0.20},
            'FWD': {KPI_PERFORMANCE: 0.15, KPI_POTENTIAL: 0.45, KPI_REGULARITY: 0.10, KPI_GOALS: 0.30}
        },
        "mrb_params_per_pos": { # More aggressive bidding for high potential might mean higher bonus
            'GK': {'max_proportional_bonus_at_pvs100': 0.5},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.6},
            'MID': {'max_proportional_bonus_at_pvs100': 0.8}, 
            'FWD': {'max_proportional_bonus_at_pvs100': 1.0}
        }
    },
    "Emphasis on Performance & Regularity": {
        "kpi_weights": {
            'GK':  {KPI_PERFORMANCE: 0.50, KPI_POTENTIAL: 0.05, KPI_REGULARITY: 0.45, KPI_GOALS: 0.00},
            'DEF': {KPI_PERFORMANCE: 0.45, KPI_POTENTIAL: 0.05, KPI_REGULARITY: 0.40, KPI_GOALS: 0.10},
            'MID': {KPI_PERFORMANCE: 0.40, KPI_POTENTIAL: 0.10, KPI_REGULARITY: 0.30, KPI_GOALS: 0.20},
            'FWD': {KPI_PERFORMANCE: 0.35, KPI_POTENTIAL: 0.10, KPI_REGULARITY: 0.25, KPI_GOALS: 0.30}
        },
        "mrb_params_per_pos": { 
            'GK': {'max_proportional_bonus_at_pvs100': 0.2},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.3},
            'MID': {'max_proportional_bonus_at_pvs100': 0.5}, 
            'FWD': {'max_proportional_bonus_at_pvs100': 0.7}
        }
    }
}

# --- Cached Data Loading and Preprocessing Function ---
@st.cache_data 
def load_and_preprocess_data(uploaded_file_obj):
    """Loads data from uploaded file and performs initial processing for new season app."""
    if uploaded_file_obj is None:
        return None
    try:
        df_input = pd.read_excel(uploaded_file_obj) if uploaded_file_obj.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded_file_obj)
        df_processed = df_input.copy()
        
        # Essential columns
        df_processed['simplified_position'] = df_processed['Poste'].apply(MPGAuctionStrategist.simplify_position)
        df_processed['player_id'] = df_processed.apply(MPGAuctionStrategist.create_player_id, axis=1)
        df_processed['Cote'] = pd.to_numeric(df_processed['Cote'], errors='coerce').fillna(1).clip(lower=1).round().astype(int)
        
        # Process new subjective KPI columns
        for kpi_col in SUBJECTIVE_KPI_COLUMNS:
            if kpi_col not in df_processed.columns:
                st.warning(f"Warning: Subjective KPI column '{kpi_col}' not found in uploaded file. Defaulting to 0.")
                df_processed[kpi_col] = 0 # Default to 0 if missing
            else:
                df_processed[kpi_col] = pd.to_numeric(df_processed[kpi_col], errors='coerce').fillna(0).clip(0, 100)
        
        # 'Indispo ?' column is optional, default to False (available) if not present
        if 'Indispo ?' not in df_processed.columns:
            df_processed['Indispo ?'] = False
        else:
            df_processed['Indispo ?'] = df_processed['Indispo ?'].astype(str).str.upper().isin(['TRUE', 'OUI', '1', 'YES', 'VRAI'])
            
        return df_processed
    except Exception as e:
        st.error(f"Error reading or initially processing file: {e}")
        return None

class MPGAuctionStrategist:
    def __init__(self): 
        self.formations = { # Can remain the same
            "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}, "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
            "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2}, "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
            "4-5-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1}, "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},
            "5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1}
        }
        self.squad_minimums = {"GK": 2, "DEF": 6, "MID": 6, "FWD": 4} # Can remain the same
        self.budget = 500 # Can remain the same

    @property
    def squad_minimums_sum_val(self): 
        return sum(self.squad_minimums.values())

    @staticmethod 
    def simplify_position(position: str) -> str: 
        if pd.isna(position) or str(position).strip() == '': return 'UNKNOWN'
        pos = str(position).upper().strip()
        if pos == 'G': return 'GK'
        elif pos in ['D', 'DL', 'DC']: return 'DEF'
        elif pos in ['M', 'MD', 'MO']: return 'MID'
        elif pos == 'A': return 'FWD'
        else: return 'UNKNOWN'

    @staticmethod 
    def create_player_id(row) -> str: 
        name = str(row.get('Joueur', '')).strip()
        simplified_pos = MPGAuctionStrategist.simplify_position(row.get('Poste', ''))
        club = str(row.get('Club', '')).strip()
        return f"{name}_{simplified_pos}_{club}"
    
    @staticmethod 
    def normalize_subjective_kpis(df: pd.DataFrame) -> pd.DataFrame: 
        """Normalizes (clips) the new subjective KPI scores to be between 0 and 100."""
        rdf = df.copy()
        for kpi_col in SUBJECTIVE_KPI_COLUMNS:
            # Assuming scores are already meant to be 0-100, just clip them.
            # If they were on different scales, more complex normalization would be needed here.
            norm_col_name = f"norm_{kpi_col}"
            rdf[norm_col_name] = rdf[kpi_col].clip(0, 100)
        return rdf

    @staticmethod 
    def calculate_pvs(df: pd.DataFrame, weights: Dict[str, Dict[str, float]]) -> pd.DataFrame: 
        """Calculates PVS based on normalized subjective KPIs and their weights."""
        rdf = df.copy()
        rdf['pvs'] = 0.0
        for pos, pos_weights in weights.items():
            mask = rdf['simplified_position'] == pos
            if not mask.any(): continue
            
            pvs_sum = pd.Series(0.0, index=rdf.loc[mask].index)
            total_weight_for_pos = sum(pos_weights.values()) # To normalize PVS if weights don't sum to 1

            for kpi_col_base_name in SUBJECTIVE_KPI_COLUMNS:
                norm_kpi_col = f"norm_{kpi_col_base_name}"
                weight = pos_weights.get(kpi_col_base_name, 0) # Get weight for this subjective KPI
                if norm_kpi_col in rdf.columns: # Ensure normalized column exists
                    pvs_sum += rdf.loc[mask, norm_kpi_col].fillna(0) * weight
            
            if total_weight_for_pos > 0: # Avoid division by zero if all weights are zero
                 rdf.loc[mask, 'pvs'] = (pvs_sum / total_weight_for_pos * 100).clip(0, 100) if total_weight_for_pos != 1 else pvs_sum.clip(0, 100)
            else:
                 rdf.loc[mask, 'pvs'] = 0.0

        return rdf

    @staticmethod 
    def calculate_mrb(df: pd.DataFrame, mrb_params_per_pos: Dict[str, Dict[str, float]]) -> pd.DataFrame: 
        """Calculates MRB. Same logic as before, using the new PVS."""
        rdf = df.copy()
        rdf['mrb'] = rdf['Cote'] 
        for pos_simplified, params in mrb_params_per_pos.items():
            mask = rdf['simplified_position'] == pos_simplified 
            if not mask.any(): continue
            max_prop_bonus = params.get('max_proportional_bonus_at_pvs100', 0.5)

            def _calc_mrb_player(row): # Renamed for clarity, logic is same
                cote = int(row['Cote'])
                pvs_player = float(row['pvs']) 
                pvs_scaled = pvs_player / 100.0
                bonus_factor = pvs_scaled * max_prop_bonus
                mrb_float = cote * (1 + bonus_factor)
                mrb_capped = min(mrb_float, float(cote * 2)) # Cap at 2x Cote
                final_mrb = max(float(cote), mrb_capped) 
                return int(round(final_mrb))

            rdf.loc[mask, 'mrb'] = rdf.loc[mask].apply(_calc_mrb_player, axis=1)
        rdf['mrb'] = rdf['mrb'].astype(int)
        safe_mrb = rdf['mrb'].replace(0, 1).astype(float) 
        rdf['value_per_cost'] = rdf['pvs'] / safe_mrb
        rdf['value_per_cost'].fillna(0, inplace=True)
        return rdf

    @staticmethod 
    @st.cache_data 
    def get_evaluated_players_df(df_processed: pd.DataFrame, 
                                 # n_recent is removed
                                 kpi_weights: Dict[str, Dict[str, float]], 
                                 mrb_params: Dict[str, Dict[str, float]]):
        """Orchestrates player evaluation for the new season app."""
        if df_processed is None or df_processed.empty:
            return pd.DataFrame()
        
        # No historical KPI calculation needed here.
        # We directly normalize the subjective KPIs from df_processed.
        df_norm_subjective_kpis = MPGAuctionStrategist.normalize_subjective_kpis(df_processed)
        df_pvs = MPGAuctionStrategist.calculate_pvs(df_norm_subjective_kpis, kpi_weights)
        df_mrb = MPGAuctionStrategist.calculate_mrb(df_pvs, mrb_params)
        return df_mrb

    # --- Squad Selection Logic (select_squad) ---
    # This method is largely the same as the optimized version from mercato_mpg_gemini.txt,
    # but the min_recent_games_played_filter_value parameter and its related filtering are removed.
    def select_squad(self, df_evaluated_players: pd.DataFrame, formation_key: str, target_squad_size: int) -> Tuple[pd.DataFrame, Dict]:
        # Removed min_recent_games_played_filter_value
        
        eligible_df_initial = df_evaluated_players.copy()
        # Removed filtering based on min_recent_games_played_count
        
        if eligible_df_initial.empty:
            return pd.DataFrame(), {}
        
        eligible_df = eligible_df_initial.drop_duplicates(subset=['player_id']).copy()
        eligible_df['mrb'] = eligible_df['mrb'].astype(int)

        current_squad_list_of_dicts: List[Dict] = [] 
        
        # Helper functions ( 그대로 유지 from previous version, they use self for formations/minimums )
        def get_current_squad_player_ids_set() -> Set[str]:
            return {p['player_id'] for p in current_squad_list_of_dicts}

        def get_current_pos_counts_dict() -> Dict[str, int]:
            counts = {pos_key: 0 for pos_key in self.squad_minimums.keys()} 
            for p_dict in current_squad_list_of_dicts:
                counts[p_dict['pos']] = counts.get(p_dict['pos'], 0) + 1
            return counts

        def add_player_to_current_squad_list(player_row_data: pd.Series, is_starter_role: bool) -> bool:
            player_id_to_add = player_row_data['player_id']
            if player_id_to_add in get_current_squad_player_ids_set(): return False 
            if player_row_data['simplified_position'] == 'GK':
                if get_current_pos_counts_dict().get('GK', 0) >= 2: return False 
            current_squad_list_of_dicts.append({
                'player_id': player_id_to_add, 'mrb': int(player_row_data['mrb']),
                'pvs': float(player_row_data['pvs']), 'pos': player_row_data['simplified_position'],
                'is_starter': is_starter_role, 'Joueur': player_row_data['Joueur'] 
            })
            return True

        def remove_player_from_current_squad_list(player_id_to_remove: str) -> bool:
            nonlocal current_squad_list_of_dicts
            initial_len = len(current_squad_list_of_dicts)
            current_squad_list_of_dicts = [p for p in current_squad_list_of_dicts if p['player_id'] != player_id_to_remove]
            return len(current_squad_list_of_dicts) < initial_len
            
        all_players_sorted_pvs = eligible_df.sort_values(by='pvs', ascending=False)

        # Phase A1: Starters
        starters_map = self.formations[formation_key].copy()
        for _, player_row in all_players_sorted_pvs.iterrows():
            pos = player_row['simplified_position']
            if player_row['player_id'] not in get_current_squad_player_ids_set() and starters_map.get(pos, 0) > 0:
                if add_player_to_current_squad_list(player_row, True):
                    starters_map[pos] -= 1
        
        # Phase A2: Minimums
        current_counts_ph_a2 = get_current_pos_counts_dict()
        for pos, min_needed in self.squad_minimums.items():
            while current_counts_ph_a2.get(pos, 0) < min_needed:
                candidate_series = all_players_sorted_pvs[
                    (all_players_sorted_pvs['simplified_position'] == pos) &
                    (~all_players_sorted_pvs['player_id'].isin(get_current_squad_player_ids_set()))].head(1)
                if candidate_series.empty: break 
                if add_player_to_current_squad_list(candidate_series.iloc[0], False):
                    current_counts_ph_a2 = get_current_pos_counts_dict() 
                else: break 

        # Phase A3: Target Squad Size
        while len(current_squad_list_of_dicts) < target_squad_size:
            current_counts_ph_a3 = get_current_pos_counts_dict()
            most_needed_pos_details = [] 
            for pos_key, num_starters in self.formations[formation_key].items():
                desired_for_pos = num_starters + 1 
                deficit = desired_for_pos - current_counts_ph_a3.get(pos_key, 0)
                most_needed_pos_details.append((pos_key, deficit, num_starters))
            most_needed_pos_details.sort(key=lambda x: (x[1], x[2]), reverse=True)
            player_added_in_a3_pass = False
            for pos_to_fill, deficit_val, _ in most_needed_pos_details:
                if deficit_val <= 0 and len(current_squad_list_of_dicts) < target_squad_size : pass 
                elif deficit_val <=0: continue
                if pos_to_fill == 'GK' and current_counts_ph_a3.get('GK', 0) >= 2: continue 
                candidate_series_a3 = all_players_sorted_pvs[(all_players_sorted_pvs['simplified_position'] == pos_to_fill) & (~all_players_sorted_pvs['player_id'].isin(get_current_squad_player_ids_set()))].head(1)
                if not candidate_series_a3.empty:
                    if add_player_to_current_squad_list(candidate_series_a3.iloc[0], False):
                        player_added_in_a3_pass = True; break 
            if not player_added_in_a3_pass: 
                if len(current_squad_list_of_dicts) < target_squad_size:
                    candidate_overall_a3 = all_players_sorted_pvs[~all_players_sorted_pvs['player_id'].isin(get_current_squad_player_ids_set())].head(1)
                    if not candidate_overall_a3.empty:
                        if not add_player_to_current_squad_list(candidate_overall_a3.iloc[0], False):
                            all_players_sorted_pvs = all_players_sorted_pvs[all_players_sorted_pvs['player_id'] != candidate_overall_a3.iloc[0]['player_id']]
                            if all_players_sorted_pvs.empty: break 
                            continue 
                    else: break 
                else: break 
            if len(current_squad_list_of_dicts) >= target_squad_size: break

        current_total_mrb = sum(p['mrb'] for p in current_squad_list_of_dicts)
        
        # Phase B: Budget Conformance (Downgrades) - Logic remains the same
        max_budget_iterations_b = target_squad_size * 2; iterations_b_count = 0; budget_conformance_tolerance = 1 
        while current_total_mrb > self.budget + budget_conformance_tolerance and iterations_b_count < max_budget_iterations_b:
            iterations_b_count += 1; made_a_downgrade_in_pass = False
            best_downgrade_action = None 
            candidates_for_replacement = sorted([p for p in current_squad_list_of_dicts if not p['is_starter']], key=lambda x: x['mrb'], reverse=True)
            if not candidates_for_replacement and current_total_mrb > self.budget:
                candidates_for_replacement = sorted(current_squad_list_of_dicts, key=lambda x: x['pvs'])
            for old_player_dict_b in candidates_for_replacement:
                old_pid_b = old_player_dict_b['player_id']; old_pos_b = old_player_dict_b['pos']; old_mrb_b = old_player_dict_b['mrb']; old_pvs_b = old_player_dict_b['pvs']
                potential_replacements_df = eligible_df[(eligible_df['simplified_position'] == old_pos_b) & (~eligible_df['player_id'].isin(get_current_squad_player_ids_set() - {old_pid_b})) & (eligible_df['mrb'] < old_mrb_b)]
                if not potential_replacements_df.empty:
                    for _, new_player_row_b in potential_replacements_df.iterrows():
                        if new_player_row_b['simplified_position'] == 'GK':
                            is_old_gk = old_pos_b == 'GK'; current_gk_count_b = get_current_pos_counts_dict().get('GK',0)
                            if not is_old_gk and current_gk_count_b >=2: continue
                            if is_old_gk and current_gk_count_b > 2 : continue 
                        mrb_saved_b = old_mrb_b - new_player_row_b['mrb']
                        pvs_change_val_b = new_player_row_b['pvs'] - old_pvs_b 
                        current_swap_score_b = mrb_saved_b - (abs(pvs_change_val_b) * 0.5 if pvs_change_val_b < 0 else -pvs_change_val_b * 0.1) 
                        if best_downgrade_action is None or current_swap_score_b > best_downgrade_action[4]:
                            best_downgrade_action = (old_pid_b, new_player_row_b['player_id'], mrb_saved_b, pvs_change_val_b, current_swap_score_b, new_player_row_b)
            if best_downgrade_action:
                old_id_exec, new_id_exec, mrb_s_exec, pvs_c_exec, _, new_player_data_exec = best_downgrade_action
                original_starter_status_exec = next((p['is_starter'] for p in current_squad_list_of_dicts if p['player_id'] == old_id_exec), False)
                if remove_player_from_current_squad_list(old_id_exec):
                    if add_player_to_current_squad_list(new_player_data_exec, original_starter_status_exec):
                        current_total_mrb = sum(p['mrb'] for p in current_squad_list_of_dicts); made_a_downgrade_in_pass = True
                    else: old_player_original_data = eligible_df[eligible_df['player_id'] == old_id_exec].iloc[0]; add_player_to_current_squad_list(old_player_original_data, original_starter_status_exec); break 
            if not made_a_downgrade_in_pass:
                if current_total_mrb > self.budget + budget_conformance_tolerance: st.warning(f"Budget Target Not Met: MRB {current_total_mrb} > Budget {self.budget}.")
                break 
        
        # Phase C: Budget Upgrades - Logic remains the same
        budget_left_for_upgrades = self.budget - current_total_mrb; max_upgrade_passes_c = target_squad_size; upgrade_pass_count_c = 0
        while budget_left_for_upgrades > 5 and upgrade_pass_count_c < max_upgrade_passes_c and len(current_squad_list_of_dicts) == target_squad_size :
            upgrade_pass_count_c += 1; made_an_upgrade_this_pass_c = False
            best_upgrade_action_c = None 
            squad_for_upgrade_cands_c = sorted([p for p in current_squad_list_of_dicts if not p['is_starter']], key=lambda x: x['pvs']) 
            if not squad_for_upgrade_cands_c: squad_for_upgrade_cands_c = sorted(current_squad_list_of_dicts, key=lambda x: x['pvs'])
            for old_player_dict_c in squad_for_upgrade_cands_c:
                old_pid_c = old_player_dict_c['player_id']; old_pos_c = old_player_dict_c['pos']; old_mrb_c = old_player_dict_c['mrb']; old_pvs_c = old_player_dict_c['pvs']
                potential_upgrades_df = eligible_df[(eligible_df['simplified_position'] == old_pos_c) & (~eligible_df['player_id'].isin(get_current_squad_player_ids_set() - {old_pid_c})) & (eligible_df['pvs'] > old_pvs_c) & (eligible_df['mrb'] > old_mrb_c)]
                for _, new_player_row_c in potential_upgrades_df.iterrows():
                    if new_player_row_c['simplified_position'] == 'GK':
                        is_old_gk_c = old_pos_c == 'GK'; current_gk_count_c = get_current_pos_counts_dict().get('GK',0)
                        if not is_old_gk_c and current_gk_count_c >=2 : continue
                        if is_old_gk_c and current_gk_count_c > 2 and new_player_row_c['player_id'] != old_pid_c: continue
                    mrb_increase_c = new_player_row_c['mrb'] - old_mrb_c
                    pvs_gain_c = new_player_row_c['pvs'] - old_pvs_c
                    if mrb_increase_c <= budget_left_for_upgrades and mrb_increase_c >=0:
                        current_upgrade_score_c = pvs_gain_c / (mrb_increase_c + 0.01) 
                        if best_upgrade_action_c is None or current_upgrade_score_c > best_upgrade_action_c[5] :
                            best_upgrade_action_c = (old_pid_c, new_player_row_c['player_id'], mrb_increase_c, pvs_gain_c, new_player_row_c, current_upgrade_score_c)
            if best_upgrade_action_c:
                old_id_exec_c, new_id_exec_c, mrb_inc_exec_c, pvs_g_exec_c, new_player_data_exec_c, _ = best_upgrade_action_c
                original_starter_status_c = next((p['is_starter'] for p in current_squad_list_of_dicts if p['player_id'] == old_id_exec_c), False)
                if remove_player_from_current_squad_list(old_id_exec_c):
                    if add_player_to_current_squad_list(new_player_data_exec_c, original_starter_status_c):
                        current_total_mrb += mrb_inc_exec_c; budget_left_for_upgrades = self.budget - current_total_mrb; made_an_upgrade_this_pass_c = True
                    else: old_player_original_data_c = eligible_df[eligible_df['player_id'] == old_id_exec_c].iloc[0]; add_player_to_current_squad_list(old_player_original_data_c, original_starter_status_c); break
                else: break
            if not made_an_upgrade_this_pass_c: break

        if not current_squad_list_of_dicts: return pd.DataFrame(), {}
        final_squad_player_ids = get_current_squad_player_ids_set()
        final_squad_df_base = eligible_df[eligible_df['player_id'].isin(final_squad_player_ids)].copy()
        details_df_final = pd.DataFrame(current_squad_list_of_dicts)
        details_df_final_renamed = details_df_final.rename(columns={'mrb': 'mrb_actual_cost', 'pvs':'pvs_in_squad', 'is_starter':'is_starter_from_selection'})
        final_squad_df = pd.merge(final_squad_df_base, details_df_final_renamed[['player_id', 'mrb_actual_cost', 'pvs_in_squad', 'is_starter_from_selection']], on='player_id', how='left')
        for col in ['mrb_actual_cost', 'pvs_in_squad']: # Ensure these columns exist and are correct type
            if col not in final_squad_df.columns: final_squad_df[col] = 0
        final_squad_df['mrb_actual_cost'] = final_squad_df['mrb_actual_cost'].fillna(0).astype(int)
        final_squad_df['pvs_in_squad'] = final_squad_df['pvs_in_squad'].fillna(0.0)

        final_starter_ids_definitive = set()
        temp_formation_needs_final = self.formations[formation_key].copy()
        final_squad_df_sorted_for_final_starters = final_squad_df.sort_values(by='pvs_in_squad', ascending=False)
        for _, player_row_final_pass in final_squad_df_sorted_for_final_starters.iterrows():
            pos_final_pass = player_row_final_pass['simplified_position']; player_id_final_pass = player_row_final_pass['player_id']
            if temp_formation_needs_final.get(pos_final_pass, 0) > 0:
                if player_id_final_pass not in final_starter_ids_definitive :
                    final_starter_ids_definitive.add(player_id_final_pass); temp_formation_needs_final[pos_final_pass] -=1
        final_squad_df['is_starter'] = final_squad_df['player_id'].isin(final_starter_ids_definitive)
        if 'is_starter_from_selection' in final_squad_df.columns: final_squad_df.drop(columns=['is_starter_from_selection'], inplace=True, errors='ignore')

        final_total_mrb_actual = final_squad_df['mrb_actual_cost'].sum()
        summary = {'total_players': len(final_squad_df), 'total_cost': int(final_total_mrb_actual),
            'remaining_budget': int(self.budget - final_total_mrb_actual), 
            'position_counts': final_squad_df['simplified_position'].value_counts().to_dict(),
            'total_squad_pvs': round(final_squad_df['pvs_in_squad'].sum(), 2),
            'total_starters_pvs': round(final_squad_df[final_squad_df['is_starter']]['pvs_in_squad'].sum(), 2)}
        
        final_pos_counts_check_final = summary['position_counts']
        for pos_check, min_val_check in self.squad_minimums.items():
            if final_pos_counts_check_final.get(pos_check,0) < min_val_check: st.error(f"Issue: {pos_check} min not met!")
        if len(final_squad_df) != target_squad_size : st.error(f"Issue: Final squad size {len(final_squad_df)} != target {target_squad_size}.")
        if final_pos_counts_check_final.get('GK',0) > 2: st.error(f"Issue: Too many GKs! ({final_pos_counts_check_final.get('GK',0)}/2)")
        return final_squad_df, summary

# --- Main Streamlit App UI ---
def main():
    st.markdown('<h1 class="main-header">🚀 MPG New Season Strategist</h1>', unsafe_allow_html=True)
    strategist = MPGAuctionStrategist() 

    # --- Session State Initialization ---
    if 'current_profile_name' not in st.session_state:
        st.session_state.current_profile_name = "Balanced Value" 
        profile_values = PREDEFINED_PROFILES[st.session_state.current_profile_name]
        # Removed n_recent and min_recent_filter from session state init for profiles
        st.session_state.kpi_weights = profile_values.get("kpi_weights", PREDEFINED_PROFILES["Balanced Value"]["kpi_weights"]) 
        st.session_state.mrb_params_per_pos = profile_values.get("mrb_params_per_pos", PREDEFINED_PROFILES["Balanced Value"]["mrb_params_per_pos"])
    
    if 'formation_key' not in st.session_state: st.session_state.formation_key = DEFAULT_FORMATION 
    if 'squad_size' not in st.session_state: st.session_state.squad_size = DEFAULT_SQUAD_SIZE

    # --- Sidebar UI ---
    st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFr1EWtMR2tHq1FwHnCHqg2uXv1JMLYQlRZw&s", width=100)
    st.sidebar.markdown('<h2 class="section-header" style="margin-top:0;">⚙️ Controls & Settings</h2>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("📁 Upload Player File (CSV/Excel)", type=['csv', 'xlsx', 'xls'], 
                                             help="Expected columns: Joueur, Poste, Club, Cote, PerformanceEstimation, PotentialEstimation, RegularityEstimation, GoalsEstimation")

    # Removed "Global Data & Form Parameters" section for n_recent and min_recent_filter
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 👥 Squad Building Parameters") 
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
    profile_names = list(PREDEFINED_PROFILES.keys()) 

    def apply_profile_settings(profile_name): 
        st.session_state.current_profile_name = profile_name
        if profile_name != "Custom" and profile_name in PREDEFINED_PROFILES:
            profile = PREDEFINED_PROFILES[profile_name]
            # Removed n_recent and min_recent_filter from profile application
            st.session_state.kpi_weights = profile.get("kpi_weights", {}).copy() 
            st.session_state.mrb_params_per_pos = profile.get("mrb_params_per_pos", {}).copy()

    selected_profile_name_ui = st.sidebar.selectbox("Select Profile", options=profile_names, 
                                                    index=profile_names.index(st.session_state.current_profile_name), 
                                                    key="profile_selector_new_season", 
                                                    help="Loads predefined settings. Modifying details below sets to 'Custom'.") 
    if selected_profile_name_ui != st.session_state.current_profile_name: 
        apply_profile_settings(selected_profile_name_ui)
        st.rerun() 

    # --- Updated KPI Weights Section for New Subjective KPIs ---
    with st.sidebar.expander("📊 Subjective KPI Weights (Click to Customize)", expanded=(st.session_state.current_profile_name == "Custom")):
        active_kpi_weights = st.session_state.kpi_weights 
        weights_ui = {} 
        for pos_key in ['GK', 'DEF', 'MID', 'FWD']: 
            st.markdown(f'<h6>{pos_key}</h6>', unsafe_allow_html=True)
            # Ensure a default structure for weights if a position is missing in the profile
            default_pos_w_structure = PREDEFINED_PROFILES["Balanced Value"]["kpi_weights"].get(pos_key, {kpi: 0.25 for kpi in SUBJECTIVE_KPI_COLUMNS})
            current_pos_w_vals = active_kpi_weights.get(pos_key, default_pos_w_structure)
            
            pos_weights_temp = {}
            for kpi_key, kpi_label in zip(
                SUBJECTIVE_KPI_COLUMNS,
                ["Performance Est.", "Potential Est.", "Regularity Est.", "Goals Est."]):
                
                # Disable Goals Estimation for GK by default, or handle it contextually
                is_disabled = (kpi_key == KPI_GOALS and pos_key == 'GK') 
                default_weight = 0.0 if is_disabled else current_pos_w_vals.get(kpi_key, 0.25)
                
                pos_weights_temp[kpi_key] = st.slider(
                    f"{kpi_label}", 0.0, 1.0, 
                    float(default_weight), 0.01, 
                    key=f"{pos_key}_{kpi_key}_weight_new",
                    disabled=is_disabled
                )
            weights_ui[pos_key] = pos_weights_temp

        if weights_ui != active_kpi_weights: 
            st.session_state.current_profile_name = "Custom" 
            st.session_state.kpi_weights = weights_ui 

    # MRB Parameters section remains the same
    with st.sidebar.expander("💰 MRB Parameters (Click to Customize)", expanded=(st.session_state.current_profile_name == "Custom")):
        active_mrb_params = st.session_state.mrb_params_per_pos
        mrb_params_ui = {} 
        for pos_key in ['GK', 'DEF', 'MID', 'FWD']:
            st.markdown(f'<h6>{pos_key}</h6>', unsafe_allow_html=True)
            default_pos_mrb_structure = PREDEFINED_PROFILES["Balanced Value"]["mrb_params_per_pos"].get(pos_key, {'max_proportional_bonus_at_pvs100': 0.2})
            current_pos_mrb_vals = active_mrb_params.get(pos_key, default_pos_mrb_structure)
            mrb_params_ui[pos_key] = {
                'max_proportional_bonus_at_pvs100': st.slider(f"Max Bonus Factor (at PVS 100)", 0.0, 1.0, 
                                                              float(current_pos_mrb_vals.get('max_proportional_bonus_at_pvs100', 0.2)), 
                                                              0.01, key=f"{pos_key}_mrbMPB_new_season", 
                                                              help="Bonus factor if PVS=100. Overall MRB is capped at 2x Cote.")
            }
        if mrb_params_ui != active_mrb_params:
            st.session_state.current_profile_name = "Custom"
            st.session_state.mrb_params_per_pos = mrb_params_ui 
    
    # --- Main Panel: Calculation and Display ---
    if uploaded_file:
        df_processed_calc = load_and_preprocess_data(uploaded_file) 

        if df_processed_calc is not None and not df_processed_calc.empty:
            with st.spinner("🧠 Calculating player evaluations..."):
                try:
                    df_evaluated_players = MPGAuctionStrategist.get_evaluated_players_df( 
                        df_processed_calc, 
                        st.session_state.kpi_weights, 
                        st.session_state.mrb_params_per_pos
                    )
                    
                    if not df_evaluated_players.empty:
                        with st.spinner("🎯 Selecting optimal squad..."):
                            squad_df_result, squad_summary_result = strategist.select_squad( 
                                df_evaluated_players, 
                                st.session_state.formation_key, 
                                st.session_state.squad_size
                                # min_recent_filter removed
                            )
                        
                            st.session_state['df_for_display_new_season'] = df_evaluated_players 
                            st.session_state['squad_df_result_new_season'] = squad_df_result
                            st.session_state['squad_summary_result_new_season'] = squad_summary_result 
                            st.session_state['selected_formation_key_new_season'] = st.session_state.formation_key
                    else:
                        st.warning("No players available after evaluation. Check input data and subjective scores.")
                        for key in ['df_for_display_new_season', 'squad_df_result_new_season', 'squad_summary_result_new_season']:
                            if key in st.session_state: del st.session_state[key]
                except Exception as e:
                    st.error(f"💥 Error during calculation pipeline: {str(e)}")
                    # st.exception(e) # For debugging

        # --- Display Logic ---
        if 'squad_df_result_new_season' in st.session_state and \
           st.session_state['squad_df_result_new_season'] is not None and \
           not st.session_state['squad_df_result_new_season'].empty: 
            
            col_main_results, col_summary = st.columns([3, 1]) 
            with col_main_results:
                st.markdown('<h2 class="section-header">🏆 Suggested Squad</h2>', unsafe_allow_html=True)
                sdf = st.session_state['squad_df_result_new_season'].copy()
                
                # Columns to display in squad view (adjust based on new KPIs)
                squad_display_cols = ['Joueur', 'Club', 'simplified_position', 'pvs_in_squad', 'Cote', 'mrb_actual_cost', 
                                      KPI_PERFORMANCE, KPI_POTENTIAL, KPI_REGULARITY, KPI_GOALS, # Show raw subjective scores
                                      'value_per_cost', 'is_starter']
                
                squad_cols_exist_display = [col for col in squad_display_cols if col in sdf.columns]
                sdf_display = sdf[squad_cols_exist_display].copy() # Use a new df for display modifications

                for col in ['mrb_actual_cost', 'Cote'] + SUBJECTIVE_KPI_COLUMNS: # Ensure numeric display for KPIs
                    if col in sdf_display.columns: 
                        sdf_display[col] = pd.to_numeric(sdf_display[col], errors='coerce').fillna(0).round(0).astype(int)
                
                sdf_display.rename(columns={ 
                    'Joueur': 'Player', 'simplified_position': 'Pos', 'pvs_in_squad': 'PVS', 
                    'Cote': 'Cote', 'mrb_actual_cost': 'Suggested Bid', 
                    KPI_PERFORMANCE: 'Perf.Est.', KPI_POTENTIAL: 'Pot.Est.',
                    KPI_REGULARITY: 'Reg.Est.', KPI_GOALS: 'Goals.Est.',
                    'value_per_cost': 'Val/MRB', 'is_starter': 'Starter' 
                }, inplace=True)
                
                for col in ['PVS', 'Val/MRB']: 
                    if col in sdf_display.columns: 
                        sdf_display[col] = pd.to_numeric(sdf_display[col], errors='coerce').fillna(0.0).round(2)
                
                pos_order = ['GK', 'DEF', 'MID', 'FWD']
                if 'Pos' in sdf_display.columns:
                    sdf_display['Pos'] = pd.Categorical(sdf_display['Pos'], categories=pos_order, ordered=True) 
                    sdf_display = sdf_display.sort_values(by=['Starter', 'Pos', 'PVS'], ascending=[False, True, False])
                st.dataframe(sdf_display, use_container_width=True, hide_index=True)

            with col_summary: 
                st.markdown('<h2 class="section-header">📈 Squad Summary</h2>', unsafe_allow_html=True)
                summary = st.session_state['squad_summary_result_new_season']
                if summary and isinstance(summary, dict):
                    st.metric("Budget Spent (MRB)", f"€ {summary.get('total_cost', 0):.0f}", help=f"Remaining: € {summary.get('remaining_budget', 0):.0f}") 
                    st.metric("Squad Size", f"{summary.get('total_players', 0)} (Target: {st.session_state.squad_size})") 
                    st.metric("Total Squad PVS", f"{summary.get('total_squad_pvs', 0):.2f}")
                    st.metric("Starters PVS", f"{summary.get('total_starters_pvs', 0):.2f}")
                    st.info(f"**Formation:** {st.session_state.get('selected_formation_key_new_season', 'N/A')}")
                    st.markdown("**Positional Breakdown:**") 
                    for pos_cat_sum in pos_order: 
                        count_sum = summary.get('position_counts', {}).get(pos_cat_sum, 0)
                        min_req_sum = strategist.squad_minimums.get(pos_cat_sum, 0) 
                        st.markdown(f"• **{pos_cat_sum}:** {count_sum} (Min: {min_req_sum})") 
                else: 
                    st.warning("Squad summary unavailable.")
            
            st.markdown('<hr><h2 class="section-header">📋 Full Player Database & Values</h2>', unsafe_allow_html=True) 
            if 'df_for_display_new_season' in st.session_state and st.session_state['df_for_display_new_season'] is not None:
                df_full = st.session_state['df_for_display_new_season'].copy()
                
                # Columns for full display
                full_display_cols = ['Joueur', 'Club', 'simplified_position', 'pvs', 'Cote', 'mrb', 'Indispo ?'] + \
                                    SUBJECTIVE_KPI_COLUMNS + \
                                    [f'norm_{kpi}' for kpi in SUBJECTIVE_KPI_COLUMNS] + \
                                    ['value_per_cost']
                full_cols_exist = [col for col in full_display_cols if col in df_full.columns]
                df_full_display = df_full[full_cols_exist].copy()

                for col in ['Cote', 'mrb'] + SUBJECTIVE_KPI_COLUMNS:
                    if col in df_full_display.columns:
                        df_full_display[col] = pd.to_numeric(df_full_display[col], errors='coerce').fillna(0).round(0).astype(int)
                
                df_full_display.rename(columns={ 
                    'Joueur': 'Player', 'simplified_position': 'Pos', 'pvs': 'PVS', 'Cote': 'Cote', 
                    'mrb': 'Suggested Bid', 'Indispo ?': 'Unavail.',
                    KPI_PERFORMANCE: 'Perf.Est.', KPI_POTENTIAL: 'Pot.Est.',
                    KPI_REGULARITY: 'Reg.Est.', KPI_GOALS: 'Goals.Est.',
                    f'norm_{KPI_PERFORMANCE}': 'N.Perf', f'norm_{KPI_POTENTIAL}': 'N.Pot',
                    f'norm_{KPI_REGULARITY}': 'N.Reg', f'norm_{KPI_GOALS}': 'N.Goals',
                    'value_per_cost': 'Val/MRB'
                }, inplace=True)
                
                for col in ['PVS', 'Val/MRB'] + [f'N.{kpi[:3]}' for kpi in ['Perf','Pot','Reg','Goals'] if f'N.{kpi[:3]}' in df_full_display.columns]:
                    if col in df_full_display.columns:
                        df_full_display[col] = pd.to_numeric(df_full_display[col], errors='coerce').fillna(0.0).round(2)

                search_all = st.text_input("🔍 Search All Players:", key="search_all_new_season") 
                if search_all: 
                    df_full_display = df_full_display[df_full_display.apply(lambda r: r.astype(str).str.contains(search_all, case=False, na=False).any(), axis=1)]
                st.dataframe(df_full_display.sort_values(by='PVS', ascending=False), use_container_width=True, hide_index=True, height=600)
                st.download_button( 
                    label="📥 Download Full Analysis (CSV)", 
                    data=df_full_display.to_csv(index=False).encode('utf-8'),
                    file_name="mpg_new_season_analysis.csv", 
                    mime="text/csv",
                    key="download_new_season" )
            elif 'squad_df_result_new_season' not in st.session_state and uploaded_file: 
                st.info("📊 Adjust settings or check file. Results update dynamically.") 
        
        else: # No results to display, but file might have been uploaded and processed (e.g. empty df)
            if uploaded_file and (st.session_state.get('df_for_display_new_season') is None or (hasattr(st.session_state.get('df_for_display_new_season'), 'empty') and st.session_state.get('df_for_display_new_season').empty)):
                st.warning("Data processed, but no players available. Check your uploaded file content and subjective scores.")
            else: # No file uploaded
                st.info("👈 Upload your Player File to begin.") 
                st.markdown('<hr><h2 class="section-header">📋 Expected File Format</h2>', unsafe_allow_html=True)
                example_data_new = {
                    'Joueur': ['New Star A', 'Solid Mid B', 'Young GK C'], 
                    'Poste': ['A', 'M', 'G'], 
                    'Club': ['Team X', 'Team Y', 'Team Z'], 
                    'Cote': [20, 12, 5], 
                    KPI_PERFORMANCE: [70, 75, 60], 
                    KPI_POTENTIAL: [85, 65, 80], 
                    KPI_REGULARITY: [70, 85, 90], 
                    KPI_GOALS: [70, 40, 0] 
                }
                st.dataframe(pd.DataFrame(example_data_new), use_container_width=True, hide_index=True)
                st.markdown(f"""
                **Required Columns:** `Joueur`, `Poste`, `Club`, `Cote`
                **Subjective Score Columns (0-100 scale recommended):** `{KPI_PERFORMANCE}`, `{KPI_POTENTIAL}`, `{KPI_REGULARITY}`, `{KPI_GOALS}`
                Optional: `Indispo ?` (for unavailable players)
                """)

if __name__ == "__main__":
    main()
