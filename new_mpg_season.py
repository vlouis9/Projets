import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Set

# Page configuration
st.set_page_config(
    page_title="MPG Auction Strategist v5 (with Squad Saving)",
    page_icon="💾",
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

# Constants and Predefined Profiles
DEFAULT_SQUAD_SIZE = 20
DEFAULT_FORMATION = "4-4-2"

PREDEFINED_PROFILES = {
    "Custom": "custom",
    "Balanced Value": {
        "kpi_weights": {
            'GK': {'estimated_avg': 0.70, 'estimated_regularity': 0.25, 'estimated_ goals': 0.0, 'estimated_potential' : 0.05},
            'DEF': {'estimated_avg': 0.50, 'estimated_regularity': 0.30, 'estimated_ goals': 0.0, 'estimated_potential' : 0.20},
            'MID': {'estimated_avg': 0.30, 'estimated_regularity': 0.15, 'estimated_ goals': 0.35, 'estimated_potential' : 0.20},
            'FWD': {'estimated_avg': 0.20, 'estimated_regularity': 0.10, 'estimated_ goals': 0.50, 'estimated_potential' : 0.20}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.3}, 'DEF': {'max_proportional_bonus_at_pvs100': 0.4},
            'MID': {'max_proportional_bonus_at_pvs100': 0.6}, 'FWD': {'max_proportional_bonus_at_pvs100': 0.8}
        }
    },
    "Aggressive Bids (Pay for PVS)": {
        "kpi_weights": {
            'GK': {'estimated_avg': 0.35, 'estimated_regularity': 0.20, 'estimated_ goals': 0.0, 'estimated_potential' : 0.05},
            'DEF': {'estimated_avg': 0.30, 'estimated_regularity': 0.30, 'estimated_ goals': 0.0, 'estimated_potential' : 0.05},
            'MID': {'estimated_avg': 0.25, 'estimated_regularity': 0.15, 'estimated_ goals': 0.15, 'estimated_potential' : 0.05},
            'FWD': {'estimated_avg': 0.25, 'estimated_regularity': 0.10, 'estimated_ goals': 0.20, 'estimated_potential' : 0.05}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 1.1}, 'DEF': {'max_proportional_bonus_at_pvs100': 0.9},
            'MID': {'max_proportional_bonus_at_pvs100': 1.1}, 'FWD': {'max_proportional_bonus_at_pvs100': 1.5}
        }
    },
    "Focus on Recent Form": {
        "kpi_weights": {
            'GK': {'estimated_avg': 0.1, 'estimated_regularity': 0.3, 'estimated_ goals': 0.0, 'estimated_potential' : 0.05},
            'DEF': {'estimated_avg': 0.1, 'estimated_regularity': 0.4, 'estimated_ goals': 0.0, 'estimated_potential' : 0.05},
            'MID': {'estimated_avg': 0.1, 'estimated_regularity': 0.15, 'estimated_ goals': 0.1, 'estimated_potential' : 0.05},
            'FWD': {'estimated_avg': 0.1, 'estimated_regularity': 0.1, 'estimated_ goals': 0.1, 'estimated_potential' : 0.05}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.6}, 'DEF': {'max_proportional_bonus_at_pvs100': 0.5},
            'MID': {'max_proportional_bonus_at_pvs100': 0.6}, 'FWD': {'max_proportional_bonus_at_pvs100': 0.9}
        }
    },
    "Focus on Season Consistency": {
        "kpi_weights": {
            'GK': {'estimated_avg': 0.75, 'estimated_regularity': 0.25, 'estimated_ goals': 0.0, 'estimated_potential' : 0.00},
            'DEF': {'estimated_avg': 0.75, 'estimated_regularity': 0.15, 'estimated_ goals': 0.10, 'estimated_potential' : 0.00},
            'MID': {'estimated_avg': 0.6, 'estimated_regularity': 0.1, 'estimated_ goals': 0.3, 'estimated_potential' : 0.00},
            'FWD': {'estimated_avg': 0.5, 'estimated_regularity': 0.1, 'estimated_ goals': 0.4, 'estimated_potential' : 0.00}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.9}, 'DEF': {'max_proportional_bonus_at_pvs100': 0.8},
            'MID': {'max_proportional_bonus_at_pvs100': 0.9}, 'FWD': {'max_proportional_bonus_at_pvs100': 1.2}
        }
    }
}

#File historical
@st.cache_data
def load_and_preprocess_data(uploaded_file_obj):
    if uploaded_file_obj is None: return None
    try:
        df_input = pd.read_excel(uploaded_file_obj) if uploaded_file_obj.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded_file_obj)
        df_processed = df_input.copy()
        df_processed['simplified_position'] = df_processed['Poste'].apply(MPGAuctionStrategist.simplify_position)
        df_processed['player_id'] = df_processed.apply(MPGAuctionStrategist.create_player_id, axis=1)
        df_processed['Cote'] = pd.to_numeric(df_processed['Cote'], errors='coerce').fillna(1).clip(lower=1).round().astype(int)
        if 'Indispo ?' not in df_processed.columns:
            df_processed['Indispo ?'] = False
        else:
            df_processed['Indispo ?'] = df_processed['Indispo ?'].astype(str).str.upper().isin(['TRUE', 'OUI', '1', 'YES', 'VRAI'])
        return df_processed
    except Exception as e:
        st.error(f"Error reading or initially processing file: {e}")
        return None

#File new season
@st.cache_data
def load_and_preprocess_new_data(uploaded_file_new_obj):
    if uploaded_file_new_obj is None: return None
    try:
        df_input_new = pd.read_excel(uploaded_file_new_obj) if uploaded_file_new_obj.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded_file_new_obj)
        df_processed_new = df_input_new.copy()
        df_processed_new['simplified_position'] = df_processed_new['Poste'].apply(MPGAuctionStrategist.simplify_position)
        df_processed_new['player_id'] = df_processed_new.apply(MPGAuctionStrategist.create_player_id, axis=1)
        df_processed_new['Cote'] = pd.to_numeric(df_processed_new['Cote'], errors='coerce').fillna(1).clip(lower=1).round().astype(int)
        return df_processed_new
    except Exception as e:
        st.error(f"Error reading or initially processing file: {e}")
        return None

class MPGAuctionStrategist:
    def __init__(self):
        self.formations = {
            "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}, "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
            "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2}, "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
            "4-5-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1}, "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},
            "5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1}
        }
        self.squad_minimums = {"GK": 2, "DEF": 6, "MID": 6, "FWD": 4}
        self.budget = 500

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
    def extract_rating_goals_starter(rating_str) -> Tuple[Optional[float], int, bool, bool]:
        if pd.isna(rating_str) or str(rating_str).strip() == '' or str(rating_str).strip() == '0':
            return None, 0, False, False
        val_str = str(rating_str).strip()
        goals = val_str.count('*')
        is_starter = '(' not in val_str
        clean_rating_str = re.sub(r'[()\*]', '', val_str)
        try:
            rating = float(clean_rating_str)
            return rating, goals, True, is_starter
        except ValueError:
            return None, 0, False, False

    @staticmethod
    def get_gameweek_columns(df_columns: List[str]) -> List[str]:
        gw_cols_data = [{'name': col, 'number': int(match.group(1))} for col in df_columns if (match := re.fullmatch(r'D(\d+)', col))]
        return [col['name'] for col in sorted(gw_cols_data, key=lambda x: x['number'])]
    
    @staticmethod
    def calculate_kpis(df: pd.DataFrame) -> pd.DataFrame:
        rdf = df.copy()
        all_gws = MPGAuctionStrategist.get_gameweek_columns(df.columns)
        rdf[['estimated_avg_rating','estimated_potential_rating']] = 0.0
        rdf[['estimated_ goals',
             'estimated_regularity_pct', 'total_season_gws_considered']] = 0
        for idx, row in rdf.iterrows():
            s_ratings_p, s_goals_t, s_played = [], 0, 0
            for gw_col in all_gws:
                r, g, played, starter = MPGAuctionStrategist.extract_rating_goals_starter(row.get(gw_col))
                if played and r is not None:
                    s_ratings_p.append(r); s_goals_t += g; s_played += 1
            rdf.at[idx, 'estimated_avg_rating'] = np.mean(s_ratings_p) if s_ratings_p else 0.0
            rdf.at[idx, 'estimated_potential_rating'] = np.mean(sorted(s_ratings_p, reverse=True)[:5]) if s_ratings_p else 0.0
            rdf.at[idx, 'estimated_ goals'] = s_goals_t
            rdf.at[idx, 'total_season_gws_considered'] = len(all_gws)
            rdf.at[idx, 'estimated_regularity_pct'] = (s_played / len(all_gws) * 100) if len(all_gws) > 0 else 0.0
        for col in ['estimated_ goals', 'total_season_gws_considered']:
            rdf[col] = rdf[col].astype(int)
        return rdf

    @staticmethod
    def normalize_kpis(df: pd.DataFrame) -> pd.DataFrame:
        rdf = df.copy()
        rdf['norm_estimated_avg'] = np.clip(rdf['estimated_avg_rating'] * 10, 0, 100)
        rdf['norm_estimated_potential'] = np.clip(rdf['estimated_potential_rating'] * 10, 0, 100)
        if '%Titu' in rdf.columns:
             rdf['norm_regularity_file'] = pd.to_numeric(rdf['%Titu'], errors='coerce').fillna(0).clip(0, 100)
        else:
             rdf['norm_regularity_file'] = 0
        rdf['norm_estimated_regularity'] = rdf['estimated_regularity_pct'].clip(0, 100)
        rdf[['norm_estimated_ goals']] = 0.0
        for pos in ['DEF', 'MID', 'FWD']:
            mask = rdf['simplified_position'] == pos
            if mask.any():
                max_sg = rdf.loc[mask, 'estimated_ goals'].max() if not rdf.loc[mask, 'estimated_ goals'].empty else 0
                rdf.loc[mask, 'norm_estimated_ goals'] = np.clip((rdf.loc[mask, 'estimated_ goals'] / max_sg * 100) if max_sg > 0 else 0, 0, 100)
        return rdf

    @staticmethod
    def calculate_pvs(df: pd.DataFrame, weights: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        rdf = df.copy()
        rdf['pvs'] = 0.0
        for pos, w in weights.items():
            mask = rdf['simplified_position'] == pos
            if not mask.any(): continue
            pvs_sum = pd.Series(0.0, index=rdf.loc[mask].index)
            pvs_sum += rdf.loc[mask, 'norm_estimated_avg'].fillna(0) * w.get('estimated_avg', 0)
            pvs_sum += rdf.loc[mask, 'norm_estimated_potential'].fillna(0) * w.get('estimated_potential', 0)
            pvs_sum += rdf.loc[mask, 'norm_estimated_regularity'].fillna(0) * w.get('estimated_regularity', 0)
            pvs_sum += rdf.loc[mask, 'norm_team_ranking'].fillna(0) * w.get('team_ranking', 0)
            if 'norm_regularity_file' in rdf.columns:
                 pvs_sum += rdf.loc[mask, 'norm_regularity_file'].fillna(0) * w.get('regularity_file', 0)
            if pos in ['DEF', 'MID', 'FWD']:
                pvs_sum += rdf.loc[mask, 'norm_estimated_ goals'].fillna(0) * w.get('estimated_ goals', 0)
            rdf.loc[mask, 'pvs'] = pvs_sum.clip(0, 100)
        return rdf

    @staticmethod
    def calculate_mrb(df: pd.DataFrame, mrb_params_per_pos: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        rdf = df.copy()
        rdf['mrb'] = rdf['Cote']
        for pos_simplified, params in mrb_params_per_pos.items():
            mask = rdf['simplified_position'] == pos_simplified
            if not mask.any(): continue
            max_prop_bonus = params.get('max_proportional_bonus_at_pvs100', 0.5)
            def _calc_mrb_player_v3(row):
                cote = int(row['Cote']); pvs_player_0_100 = float(row['pvs'])
                pvs_scaled_0_1 = pvs_player_0_100 / 100.0
                pvs_derived_bonus_factor = pvs_scaled_0_1 * max_prop_bonus
                mrb_float = cote * (1 + pvs_derived_bonus_factor)
                mrb_capped_at_2x_cote = min(mrb_float, float(cote * 2))
                final_mrb = max(float(cote), mrb_capped_at_2x_cote)
                return int(round(final_mrb))
            rdf.loc[mask, 'mrb'] = rdf.loc[mask].apply(_calc_mrb_player_v3, axis=1)
        rdf['mrb'] = rdf['mrb'].astype(int)
        safe_mrb = rdf['mrb'].replace(0, 1).astype(float)
        rdf['value_per_cost'] = rdf['pvs'] / safe_mrb
        rdf['value_per_cost'].fillna(0, inplace=True)
        return rdf

    @staticmethod
    @st.cache_data
    def get_evaluated_players_df(df_processed: pd.DataFrame, n_recent: int, kpi_weights: Dict[str, Dict[str, float]], mrb_params: Dict[str, Dict[str, float]]):
        if df_processed is None or df_processed.empty: return pd.DataFrame()
        df_kpis = MPGAuctionStrategist.calculate_kpis(df_processed, n_recent)
        df_norm_kpis = MPGAuctionStrategist.normalize_kpis(df_kpis)
        df_pvs = MPGAuctionStrategist.calculate_pvs(df_norm_kpis, kpi_weights)
        df_mrb = MPGAuctionStrategist.calculate_mrb(df_pvs, mrb_params)
        return df_mrb

    def select_squad(self, df_evaluated_players: pd.DataFrame, formation_key: str, target_squad_size: int, min_recent_games_played_filter_value: int) -> Tuple[pd.DataFrame, Dict]:
        eligible_df_initial = df_evaluated_players.copy()
        if min_recent_games_played_filter_value > 0:
            eligible_df_initial = eligible_df_initial[eligible_df_initial['recent_games_played_count'] >= min_recent_games_played_filter_value]
        if eligible_df_initial.empty: return pd.DataFrame(), {}
        eligible_df = eligible_df_initial.drop_duplicates(subset=['player_id']).copy()
        eligible_df['mrb'] = eligible_df['mrb'].astype(int)
        current_squad_list_of_dicts: List[Dict] = []
        def get_current_squad_player_ids_set() -> Set[str]: return {p['player_id'] for p in current_squad_list_of_dicts}
        def get_current_pos_counts_dict() -> Dict[str, int]:
            counts = {pos_key: 0 for pos_key in self.squad_minimums.keys()}
            for p_dict in current_squad_list_of_dicts: counts[p_dict['pos']] = counts.get(p_dict['pos'], 0) + 1
            return counts
        def add_player_to_current_squad_list(player_row_data: pd.Series, is_starter_role: bool) -> bool:
            player_id_to_add = player_row_data['player_id']
            if player_id_to_add in get_current_squad_player_ids_set(): return False
            if player_row_data['simplified_position'] == 'GK':
                if get_current_pos_counts_dict().get('GK', 0) >= 2: return False
            current_squad_list_of_dicts.append({'player_id': player_id_to_add, 'mrb': int(player_row_data['mrb']),'pvs': float(player_row_data['pvs']), 'pos': player_row_data['simplified_position'],'is_starter': is_starter_role, 'Joueur': player_row_data['Joueur']})
            return True
        def remove_player_from_current_squad_list(player_id_to_remove: str) -> bool:
            nonlocal current_squad_list_of_dicts; initial_len = len(current_squad_list_of_dicts)
            current_squad_list_of_dicts = [p for p in current_squad_list_of_dicts if p['player_id'] != player_id_to_remove]
            return len(current_squad_list_of_dicts) < initial_len
        all_players_sorted_pvs = eligible_df.sort_values(by='pvs', ascending=False)
        starters_map = self.formations[formation_key].copy()
        for _, player_row in all_players_sorted_pvs.iterrows():
            pos = player_row['simplified_position']
            if player_row['player_id'] not in get_current_squad_player_ids_set() and starters_map.get(pos, 0) > 0:
                if add_player_to_current_squad_list(player_row, True): starters_map[pos] -= 1
        current_counts_ph_a2 = get_current_pos_counts_dict()
        for pos, min_needed in self.squad_minimums.items():
            while current_counts_ph_a2.get(pos, 0) < min_needed:
                candidate_series = all_players_sorted_pvs[(all_players_sorted_pvs['simplified_position'] == pos) & (~all_players_sorted_pvs['player_id'].isin(get_current_squad_player_ids_set()))].head(1)
                if candidate_series.empty: break
                if add_player_to_current_squad_list(candidate_series.iloc[0], False): current_counts_ph_a2 = get_current_pos_counts_dict()
                else: break
        while len(current_squad_list_of_dicts) < target_squad_size:
            current_counts_ph_a3 = get_current_pos_counts_dict(); most_needed_pos_details = []
            for pos_key, num_starters in self.formations[formation_key].items():
                desired_for_pos = num_starters + 1; deficit = desired_for_pos - current_counts_ph_a3.get(pos_key, 0)
                most_needed_pos_details.append((pos_key, deficit, num_starters))
            most_needed_pos_details.sort(key=lambda x: (x[1], x[2]), reverse=True); player_added_in_a3_pass = False
            for pos_to_fill, deficit_val, _ in most_needed_pos_details:
                if deficit_val <= 0 and len(current_squad_list_of_dicts) < target_squad_size : pass
                elif deficit_val <=0: continue
                if pos_to_fill == 'GK' and current_counts_ph_a3.get('GK', 0) >= 2: continue
                candidate_series_a3 = all_players_sorted_pvs[(all_players_sorted_pvs['simplified_position'] == pos_to_fill) & (~all_players_sorted_pvs['player_id'].isin(get_current_squad_player_ids_set()))].head(1)
                if not candidate_series_a3.empty:
                    if add_player_to_current_squad_list(candidate_series_a3.iloc[0], False): player_added_in_a3_pass = True; break
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
        max_budget_iterations_b = target_squad_size * 2; iterations_b_count = 0; budget_conformance_tolerance = 1
        while current_total_mrb > self.budget + budget_conformance_tolerance and iterations_b_count < max_budget_iterations_b:
            iterations_b_count += 1; made_a_downgrade_in_pass = False; best_downgrade_action = None
            candidates_for_replacement = sorted([p for p in current_squad_list_of_dicts if not p['is_starter']], key=lambda x: x['mrb'], reverse=True)
            if not candidates_for_replacement and current_total_mrb > self.budget: candidates_for_replacement = sorted(current_squad_list_of_dicts, key=lambda x: x['pvs'])
            for old_player_dict_b in candidates_for_replacement:
                old_pid_b = old_player_dict_b['player_id']; old_pos_b = old_player_dict_b['pos']; old_mrb_b = old_player_dict_b['mrb']; old_pvs_b = old_player_dict_b['pvs']
                potential_replacements_df = eligible_df[(eligible_df['simplified_position'] == old_pos_b) & (~eligible_df['player_id'].isin(get_current_squad_player_ids_set() - {old_pid_b})) & (eligible_df['mrb'] < old_mrb_b)]
                if not potential_replacements_df.empty:
                    for _, new_player_row_b in potential_replacements_df.iterrows():
                        if new_player_row_b['simplified_position'] == 'GK':
                            is_old_gk = old_pos_b == 'GK'; current_gk_count_b = get_current_pos_counts_dict().get('GK',0)
                            if not is_old_gk and current_gk_count_b >=2: continue
                            if is_old_gk and current_gk_count_b > 2 : continue
                        mrb_saved_b = old_mrb_b - new_player_row_b['mrb']; pvs_change_val_b = new_player_row_b['pvs'] - old_pvs_b
                        current_swap_score_b = mrb_saved_b - (abs(pvs_change_val_b) * 0.5 if pvs_change_val_b < 0 else -pvs_change_val_b * 0.1)
                        if best_downgrade_action is None or current_swap_score_b > best_downgrade_action[4]:
                            best_downgrade_action = (old_pid_b, new_player_row_b['player_id'], mrb_saved_b, pvs_change_val_b, current_swap_score_b, new_player_row_b)
            if best_downgrade_action:
                old_id_exec, new_id_exec, mrb_s_exec, pvs_c_exec, _, new_player_data_exec = best_downgrade_action
                original_starter_status_exec = next((p['is_starter'] for p in current_squad_list_of_dicts if p['player_id'] == old_id_exec), False)
                if remove_player_from_current_squad_list(old_id_exec):
                    if add_player_to_current_squad_list(new_player_data_exec, original_starter_status_exec): current_total_mrb = sum(p['mrb'] for p in current_squad_list_of_dicts); made_a_downgrade_in_pass = True
                    else: old_player_original_data = eligible_df[eligible_df['player_id'] == old_id_exec].iloc[0]; add_player_to_current_squad_list(old_player_original_data, original_starter_status_exec); break
            if not made_a_downgrade_in_pass:
                if current_total_mrb > self.budget + budget_conformance_tolerance: st.warning(f"Budget Target Not Met: MRB {current_total_mrb} > Budget {self.budget}.")
                break
        budget_left_for_upgrades = self.budget - current_total_mrb; max_upgrade_passes_c = target_squad_size; upgrade_pass_count_c = 0
        while budget_left_for_upgrades > 5 and upgrade_pass_count_c < max_upgrade_passes_c and len(current_squad_list_of_dicts) == target_squad_size :
            upgrade_pass_count_c += 1; made_an_upgrade_this_pass_c = False; best_upgrade_action_c = None
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
                    mrb_increase_c = new_player_row_c['mrb'] - old_mrb_c; pvs_gain_c = new_player_row_c['pvs'] - old_pvs_c
                    if mrb_increase_c <= budget_left_for_upgrades and mrb_increase_c >=0:
                        current_upgrade_score_c = pvs_gain_c / (mrb_increase_c + 0.01)
                        if best_upgrade_action_c is None or current_upgrade_score_c > best_upgrade_action_c[5] : best_upgrade_action_c = (old_pid_c, new_player_row_c['player_id'], mrb_increase_c, pvs_gain_c, new_player_row_c, current_upgrade_score_c)
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
        details_df_final = pd.DataFrame(current_squad_list_of_dicts); details_df_final_renamed = details_df_final.rename(columns={'mrb': 'mrb_actual_cost', 'pvs':'pvs_in_squad', 'is_starter':'is_starter_from_selection'})
        final_squad_df = pd.merge(final_squad_df_base, details_df_final_renamed[['player_id', 'mrb_actual_cost', 'pvs_in_squad', 'is_starter_from_selection']], on='player_id', how='left')
        if 'mrb_actual_cost' not in final_squad_df.columns: final_squad_df['mrb_actual_cost'] = 0
        if 'pvs_in_squad' not in final_squad_df.columns: final_squad_df['pvs_in_squad'] = 0.0
        final_squad_df['mrb_actual_cost'] = final_squad_df['mrb_actual_cost'].fillna(0).astype(int)
        final_squad_df['pvs_in_squad'] = final_squad_df['pvs_in_squad'].fillna(0.0)
        final_starter_ids_definitive = set(); temp_formation_needs_final = self.formations[formation_key].copy()
        final_squad_df_sorted_for_final_starters = final_squad_df.sort_values(by='pvs_in_squad', ascending=False)
        for _, player_row_final_pass in final_squad_df_sorted_for_final_starters.iterrows():
            pos_final_pass = player_row_final_pass['simplified_position']; player_id_final_pass = player_row_final_pass['player_id']
            if temp_formation_needs_final.get(pos_final_pass, 0) > 0:
                if player_id_final_pass not in final_starter_ids_definitive : final_starter_ids_definitive.add(player_id_final_pass); temp_formation_needs_final[pos_final_pass] -=1
        final_squad_df['is_starter'] = final_squad_df['player_id'].isin(final_starter_ids_definitive)
        if 'is_starter_from_selection' in final_squad_df.columns: final_squad_df.drop(columns=['is_starter_from_selection'], inplace=True, errors='ignore')
        final_total_mrb_actual = final_squad_df['mrb_actual_cost'].sum()
        summary = {'total_players': len(final_squad_df), 'total_cost': int(final_total_mrb_actual),'remaining_budget': int(self.budget - final_total_mrb_actual), 'position_counts': final_squad_df['simplified_position'].value_counts().to_dict(),'total_squad_pvs': round(final_squad_df['pvs_in_squad'].sum(), 2),'total_starters_pvs': round(final_squad_df[final_squad_df['is_starter']]['pvs_in_squad'].sum(), 2)}
        final_pos_counts_check_final = summary['position_counts']
        for pos_check, min_val_check in self.squad_minimums.items():
            if final_pos_counts_check_final.get(pos_check,0) < min_val_check: st.error(f"Issue: {pos_check} min not met! ({final_pos_counts_check_final.get(pos_check,0)}/{min_val_check})")
        if len(final_squad_df) != target_squad_size : st.error(f"Issue: Final squad size {len(final_squad_df)} != target {target_squad_size}.")
        if final_pos_counts_check_final.get('GK',0) > 2: st.error(f"Issue: Too many GKs! ({final_pos_counts_check_final.get('GK',0)}/2)")
        return final_squad_df, summary

def main():
    st.markdown('<h1 class="main-header">🚀 MPG Auction Strategist v5 (with Squad Saving)</h1>', unsafe_allow_html=True)
    strategist = MPGAuctionStrategist()

    # --- INITIALIZE SESSION STATE for new features ---
    if 'saved_squads' not in st.session_state:
        st.session_state.saved_squads = []

    # --- Session state initialization for app parameters ---
    if 'current_profile_name' not in st.session_state:
        st.session_state.current_profile_name = "Balanced Value"
        profile_values = PREDEFINED_PROFILES[st.session_state.current_profile_name]
        st.session_state.kpi_weights = profile_values.get("kpi_weights", PREDEFINED_PROFILES["Balanced Value"]["kpi_weights"])
        st.session_state.mrb_params_per_pos = profile_values.get("mrb_params_per_pos", PREDEFINED_PROFILES["Balanced Value"]["mrb_params_per_pos"])
    if 'formation_key' not in st.session_state: st.session_state.formation_key = DEFAULT_FORMATION
    if 'squad_size' not in st.session_state: st.session_state.squad_size = DEFAULT_SQUAD_SIZE

    # --- Sidebar UI ---
    st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFr1EWtMR2tHq1FwHnCHqg2uXv1JMLYQlRZw&s", width=100)
    st.sidebar.markdown('<h2 class="section-header" style="margin-top:0;">⚙️ Controls & Settings</h2>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("📁 Upload MPG Ratings File (CSV/Excel)", type=['csv', 'xlsx', 'xls'], help="Joueur, Poste, Club, Cote, %Titu, Indispo?, Gameweeks (D1..D34).")
    uploaded_file_new = st.sidebar.file_uploader("📁 Upload MPG Players Database (CSV/Excel)", type=['csv', 'xlsx', 'xls'], help="Joueur, Poste, Club, Cote")
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🌎 Global Parameters")
    #Teams tier list weighting
    team_ranking: st.slider(f"Team ranking", 0.0, 1.0, float(current_pos_w_vals.get('team_ranking', 0.0)), 0.01, key=f"{pos_key}_wSC_v5_opt_main")
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 👥 Squad Building Parameters")
    formation_key_ui = st.sidebar.selectbox("Preferred Starting Formation", options=list(strategist.formations.keys()), index=list(strategist.formations.keys()).index(st.session_state.get('formation_key', DEFAULT_FORMATION)))
    target_squad_size_ui = st.sidebar.number_input("Target Total Squad Size", min_value=strategist.squad_minimums_sum_val, max_value=30, value=st.session_state.get('squad_size', DEFAULT_SQUAD_SIZE))
    if formation_key_ui != st.session_state.get('formation_key') or target_squad_size_ui != st.session_state.get('squad_size'): st.session_state.current_profile_name = "Custom"
    st.session_state.formation_key = formation_key_ui
    st.session_state.squad_size = target_squad_size_ui
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🎨 Settings Profiles")
    profile_names = list(PREDEFINED_PROFILES.keys())
    def apply_profile_settings(profile_name):
        st.session_state.current_profile_name = profile_name
        if profile_name != "Custom" and profile_name in PREDEFINED_PROFILES:
            profile = PREDEFINED_PROFILES[profile_name]
            st.session_state.kpi_weights = profile.get("kpi_weights", {}).copy()
            st.session_state.mrb_params_per_pos = profile.get("mrb_params_per_pos", {}).copy()
    selected_profile_name_ui = st.sidebar.selectbox("Select Profile", options=profile_names, index=profile_names.index(st.session_state.current_profile_name), key="profile_selector_v5_opt_main", help="Loads predefined settings. Modifying details below sets to 'Custom'.")
    if selected_profile_name_ui != st.session_state.current_profile_name:
        apply_profile_settings(selected_profile_name_ui)
        st.rerun()
    with st.sidebar.expander("📊 KPI Weights (Click to Customize)", expanded=(st.session_state.current_profile_name == "Custom")):
        active_kpi_weights = st.session_state.kpi_weights; weights_ui = {}
        for pos_key in ['GK', 'DEF', 'MID', 'FWD']:
            st.markdown(f'<h6>{pos_key}</h6>', unsafe_allow_html=True)
            default_pos_w_structure = PREDEFINED_PROFILES["Balanced Value"]["kpi_weights"][pos_key]; current_pos_w_vals = active_kpi_weights.get(pos_key, default_pos_w_structure)
            weights_ui[pos_key] = {
                'estimated_avg': st.slider(f"Season Avg Rating", 0.0, 1.0, float(current_pos_w_vals.get('estimated_avg', 0.0)), 0.01, key=f"{pos_key}_wSA_v5_opt_main"),
                'estimated_potential': st.slider(f"Potential Rating", 0.0, 1.0, float(current_pos_w_vals.get('estimated_potential', 0.0)), 0.01, key=f"{pos_key}_wSB_v5_opt_main"),
                'estimated_ goals': st.slider(f"Season Goals", 0.0, 1.0, float(current_pos_w_vals.get('estimated_ goals', 0.0)) if pos_key in ['DEF', 'MID', 'FWD'] else 0.0, 0.01, key=f"{pos_key}_wSG_v5_opt_main", disabled=pos_key not in ['DEF','MID', 'FWD']),
                'estimated_regularity': st.slider(f"Calculated Regularity", 0.0, 1.0, float(current_pos_w_vals.get('estimated_regularity', 0.0)), 0.01, key=f"{pos_key}_wCR_v5_opt_main", help="Based on starts identified in gameweek data."),
            }
        if weights_ui != active_kpi_weights: st.session_state.current_profile_name = "Custom"; st.session_state.kpi_weights = weights_ui
    with st.sidebar.expander("💰 MRB Parameters (Click to Customize)", expanded=(st.session_state.current_profile_name == "Custom")):
        active_mrb_params = st.session_state.mrb_params_per_pos; mrb_params_ui = {}
        for pos_key in ['GK', 'DEF', 'MID', 'FWD']:
            st.markdown(f'<h6>{pos_key}</h6>', unsafe_allow_html=True)
            default_pos_mrb_structure = PREDEFINED_PROFILES["Balanced Value"]["mrb_params_per_pos"][pos_key]; current_pos_mrb_vals = active_mrb_params.get(pos_key, default_pos_mrb_structure)
            mrb_params_ui[pos_key] = {'max_proportional_bonus_at_pvs100': st.slider(f"Max Bonus Factor (at PVS 100)", 0.0, 1.0, float(current_pos_mrb_vals.get('max_proportional_bonus_at_pvs100', 0.2)), 0.01, key=f"{pos_key}_mrbMPB_v5_opt_main", help="Bonus factor if PVS=100 (e.g., 0.5 = 50% bonus implies MRB up to 1.5x Cote). Overall MRB is capped at 2x Cote.")}
        if mrb_params_ui != active_mrb_params: st.session_state.current_profile_name = "Custom"; st.session_state.mrb_params_per_pos = mrb_params_ui

    if uploaded_file and uploaded_file_new:
        df_processed_calc = load_and_preprocess_data(uploaded_file)
        df_processed_calc_new = load_and_preprocess_new_data(uploaded_file_new_obj)
        if df_processed_calc is not None and not df_processed_calc.empty and df_processed_calc_new is not None and not df_processed_calc_new.empty:
            with st.spinner("🧠 Calculating player evaluations..."):
                try:
                    df_evaluated_players = MPGAuctionStrategist.get_evaluated_players_df(df_processed_calc, st.session_state.n_recent, st.session_state.kpi_weights, st.session_state.mrb_params_per_pos)
                    if not df_evaluated_players.empty:
                        with st.spinner("🎯 Selecting optimal squad..."):
                            squad_df_result, squad_summary_result = strategist.select_squad(df_evaluated_players, st.session_state.formation_key, st.session_state.squad_size, st.session_state.min_recent_filter)
                        st.session_state['df_for_display_final'] = df_evaluated_players
                        st.session_state['squad_df_result_final'] = squad_df_result
                        st.session_state['squad_summary_result_final'] = squad_summary_result
                        st.session_state['selected_formation_key_display_final'] = st.session_state.formation_key
                    else:
                        st.warning("No players left after evaluation. Check filters or data.")
                        for key in ['df_for_display_final', 'squad_df_result_final', 'squad_summary_result_final']:
                            if key in st.session_state: del st.session_state[key]
                except Exception as e:
                    st.error(f"💥 Error during calculation pipeline: {str(e)}")

        #Definition team ranking
        if 'team_tiers' not in st.session_state: st.session_state.team_tiers = {t: [] for t in ["Winner", "European", "Average", "Relegation"]}
    
        st.markdown('<h2 class="section-header">1. Team Ranking Setup</h2>', unsafe_allow_html=True)
        all_clubs = sorted(df_processed_new['Club'].unique())
        tier_names = ["Winner", "European", "Average", "Relegation"]
        tier_cols = st.columns(len(tier_names))
        
        assigned_clubs = {club for tier_list in st.session_state.team_tiers.values() for club in tier_list}
        for i, tier in enumerate(tier_names):
            with tier_cols[i]:
                current_selection = [c for c in st.session_state.team_tiers.get(tier, []) if c in all_clubs]
                options_for_this_tier = sorted(list((set(all_clubs) - assigned_clubs) | set(current_selection)))
                st.session_state.team_tiers[tier] = st.multiselect(f"**{tier} Tier**", options=options_for_this_tier, default=current_selection, key=f"tier_{tier}")
        
        tier_map = {100: "Winner", 75: "European", 50: "Average", 25: "Relegation"}
        club_to_score = {club: score for score, tier in tier_map.items() for club in st.session_state.team_tiers[tier]}
        
        def display_squad_dataframe(df_to_display):
            sdf = df_to_display.copy()
            squad_cols_display = ['Joueur', 'Club', 'simplified_position', 'pvs_in_squad', 'Cote', 'mrb_actual_cost', 'estimated_avg_rating', 'estimated_potential', 'estimated_ goals', 'estimated_regularity_pct', 'value_per_cost', 'is_starter']
            squad_cols_exist_display = [col for col in squad_cols_display if col in sdf.columns]
            sdf_display = sdf[squad_cols_exist_display].rename(columns={
                'Joueur': 'Player', 'simplified_position': 'Pos', 'pvs_in_squad': 'PVS', 'Cote': 'Cote',
                'mrb_actual_cost': 'Suggested Bid', 'estimated_avg_rating': 'Average', 'estimated_potential': 'Potential', 'estimated_ goals': 'Goals',
                'estimated_regularity_pct': '% played',
                'value_per_cost': 'Val/MRB', 'is_starter': 'Starter'
            })
            for col in ['PVS', 'Average', '% played', 'Rec.AvgR', 'Val/MRB']:
                if col in sdf_display.columns: sdf_display[col] = pd.to_numeric(sdf_display[col], errors='coerce').fillna(0.0).round(2)
            pos_order = ['GK', 'DEF', 'MID', 'FWD']
            if 'Pos' in sdf_display.columns:
                sdf_display['Pos'] = pd.Categorical(sdf_display['Pos'], categories=pos_order, ordered=True)
                sdf_display = sdf_display.sort_values(by=['Starter', 'Pos', 'PVS'], ascending=[False, True, False])
            st.dataframe(sdf_display, use_container_width=True, hide_index=True)

        if 'squad_df_result_final' in st.session_state and st.session_state['squad_df_result_final'] is not None and not st.session_state['squad_df_result_final'].empty:
            col_main_results, col_summary = st.columns([3, 1])
            with col_main_results:
                st.markdown('<h2 class="section-header">🏆 Suggested Squad</h2>', unsafe_allow_html=True)
                display_squad_dataframe(st.session_state['squad_df_result_final'])
            with col_summary:
                st.markdown('<h2 class="section-header">📈 Squad Summary</h2>', unsafe_allow_html=True)
                summary = st.session_state['squad_summary_result_final']
                if summary:
                    st.metric("Budget Spent (MRB)", f"€ {summary.get('total_cost', 0):.0f}", help=f"Remaining: € {summary.get('remaining_budget', 0):.0f}")
                    st.metric("Squad Size", f"{summary.get('total_players', 0)} (Target: {st.session_state.squad_size})")
                    st.metric("Total Squad PVS", f"{summary.get('total_squad_pvs', 0):.2f}")
                    st.metric("Starters PVS", f"{summary.get('total_starters_pvs', 0):.2f}")
                    st.info(f"**Formation:** {st.session_state.get('selected_formation_key_display_final', 'N/A')}")
                    pos_order = ['GK', 'DEF', 'MID', 'FWD']
                    for pos_cat_sum in pos_order:
                        count_sum = summary.get('position_counts', {}).get(pos_cat_sum, 0)
                        min_req_sum = strategist.squad_minimums.get(pos_cat_sum, 0)
                        st.markdown(f"• **{pos_cat_sum}:** {count_sum} (Min: {min_req_sum})")

                st.markdown("---")
                st.markdown("#### 💾 Save Current Squad")
                next_squad_num = 1
                if st.session_state.saved_squads:
                    default_name_nums = [int(s['name'].split('#')[-1]) for s in st.session_state.saved_squads if s['name'].startswith(f"{st.session_state.formation_key}_{st.session_state.current_profile_name}_")]
                    if default_name_nums: next_squad_num = max(default_name_nums) + 1
                default_name = f"{st.session_state.formation_key}_{st.session_state.current_profile_name}_#{next_squad_num}"
                squad_name_input = st.text_input("Enter or confirm squad name:", value=default_name, key="squad_name_input")
                if st.button("Save Squad", key="save_squad_button"):
                    if squad_name_input:
                        if any(s['name'] == squad_name_input for s in st.session_state.saved_squads):
                            st.warning(f"A squad named '{squad_name_input}' already exists.")
                        else:
                            saved_squad_df = st.session_state['squad_df_result_final'].copy()
                            saved_squad_summary = st.session_state['squad_summary_result_final'].copy()
                            saved_squad_summary['profile_name'] = st.session_state.current_profile_name
                            saved_squad_summary['formation'] = st.session_state.get('selected_formation_key_display_final', st.session_state.formation_key)
                            st.session_state.saved_squads.append({"name": squad_name_input, "squad_df": saved_squad_df, "summary": saved_squad_summary})
                            st.success(f"Squad '{squad_name_input}' saved!")
                    else: st.warning("Please enter a name to save the squad.")

            st.markdown('<hr><h2 class="section-header">📋 Full Player Database & Values</h2>', unsafe_allow_html=True)
            if 'df_for_display_final' in st.session_state and st.session_state['df_for_display_final'] is not None:
                df_full = st.session_state['df_for_display_final'].copy()
                all_stats_cols_display = ['Joueur', 'Club', 'simplified_position', 'pvs', 'Cote', 'mrb','estimated_avg_rating', 'estimated_potential', 'estimated_ goals', 'estimated_regularity_pct', 'value_per_cost']
                df_full = df_full[[col for col in all_stats_cols_display if col in df_full.columns]]
                df_full.rename(columns={
                    'Joueur': 'Player', 'simplified_position': 'Pos', 'pvs': 'PVS', 'Cote': 'Cote',
                    'mrb': 'Suggested Bid', 'Indispo ?': 'Unavail.', 'estimated_avg_rating': 'Average', 'estimated_potential': 'Potential',
                    'estimated_ goals': 'Goals', 'estimated_regularity_pct': '% Played',
                    'value_per_cost': 'Val/MRB'
                }, inplace=True)
                for col in ['PVS', 'Average', '% Played', 'Val/MRB']:
                    if col in df_full.columns: df_full[col] = pd.to_numeric(df_full[col], errors='coerce').fillna(0.0).round(2)
                search_all = st.text_input("🔍 Search All Players:", key="search_all_v5_opt_main")
                if search_all: df_full = df_full[df_full.apply(lambda r: r.astype(str).str.contains(search_all, case=False, na=False).any(), axis=1)]

                if 'PVS' in df_full.columns:
                    st.dataframe(df_full.sort_values(by='PVS', ascending=False), use_container_width=True, hide_index=True, height=600)
                else:
                    st.warning("PVS column not available for sorting. Displaying without sorting.")
                    st.dataframe(df_full, use_container_width=True, hide_index=True, height=600)

                st.download_button(label="📥 Download Full Analysis (CSV)", data=df_full.to_csv(index=False).encode('utf-8'), file_name="mpg_full_player_analysis.csv", mime="text/csv")

        if st.session_state.saved_squads:
            st.markdown('<hr style="margin-top: 2rem; margin-bottom: 2rem;">', unsafe_allow_html=True)
            st.markdown('<h2 class="section-header">💾 My Saved Squads</h2>', unsafe_allow_html=True)
            reversed_squads = reversed(st.session_state.saved_squads)
            for i, saved_item in enumerate(reversed_squads):
                original_index = len(st.session_state.saved_squads) - 1 - i
                summary_info = (f"**{saved_item['name']}** | Cost: €{saved_item['summary']['total_cost']} | PVS: {saved_item['summary']['total_squad_pvs']:.2f} | Formation: {saved_item['summary']['formation']}")
                with st.expander(summary_info):
                    st.write(f"**Profile Used:** {saved_item['summary']['profile_name']}")
                    display_squad_dataframe(saved_item['squad_df'])
                    st.markdown("---")
                    if st.button("Delete this squad", key=f"delete_squad_{original_index}", type="primary"):
                        st.session_state.saved_squads.pop(original_index)
                        st.rerun()

        else:
            st.info("👈 Upload your MPG ratings file to begin.")
            example_data = {
                'Joueur': ['Player A', 'Player B'], 'Poste': ['A', 'M'], 'Club': ['Club X', 'Club Y'],
                'Indispo ?': ['', 'TRUE'], 'Cote': [45, 30], '%Titu': [90, 75],
                'D34': ['7.5*', '6.5'], 'D33': ['(6.0)**', '0'], 'D32': ['', '5.5*']
            }
            st.dataframe(pd.DataFrame(example_data), use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
