import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import json
from typing import Dict, List, Tuple, Optional, Set

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="MPG Auction Strategist - New Season Mode",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CUSTOM CSS ----
st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #004080; text-align: center; margin-bottom: 2rem; font-family: 'Roboto', sans-serif;}
    .section-header {font-size: 1.4rem; font-weight: bold; color: #006847; margin-top: 1.5rem; margin-bottom: 1rem; border-bottom: 2px solid #006847; padding-bottom: 0.3rem;}
    .stButton>button {background-color: #004080; color: white; font-weight: bold; border-radius: 0.3rem; padding: 0.4rem 0.8rem; border: none; width: 100%;}
    .stButton>button:hover {background-color: #003060; color: white;}
</style>
""", unsafe_allow_html=True)

# ---- CONSTANTS ----
CLUB_TIERS = {
    "Winner": 100,
    "European": 75,
    "Average": 50,
    "Relegation": 25
}
CLUB_TIERS_LABELS = list(CLUB_TIERS.keys())
NEW_PLAYER_SCORE_OPTIONS = [0, 25, 50, 75, 100]
DEFAULT_SQUAD_SIZE = 20
DEFAULT_FORMATION = "4-4-2"

PREDEFINED_PROFILES = {
    "Custom": "custom",
    "Balanced Value": {
        "kpi_weights": {
            'GK': {'estimated_performance': 0.40, 'estimated_potential': 0.30, 'estimated_regularity': 0.30, 'estimated_goals': 0.0, 'team_ranking': 0.0},
            'DEF': {'estimated_performance': 0.30, 'estimated_potential': 0.25, 'estimated_regularity': 0.25, 'estimated_goals': 0.10, 'team_ranking': 0.10},
            'MID': {'estimated_performance': 0.25, 'estimated_potential': 0.25, 'estimated_regularity': 0.20, 'estimated_goals': 0.15, 'team_ranking': 0.15},
            'FWD': {'estimated_performance': 0.20, 'estimated_potential': 0.25, 'estimated_regularity': 0.15, 'estimated_goals': 0.25, 'team_ranking': 0.15}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.3},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.4},
            'MID': {'max_proportional_bonus_at_pvs100': 0.6},
            'FWD': {'max_proportional_bonus_at_pvs100': 0.8}
        }
    },
    "Potential Focus": {
        "kpi_weights": {
            'GK': {'estimated_performance': 0.20, 'estimated_potential': 0.60, 'estimated_regularity': 0.20, 'estimated_goals': 0.0, 'team_ranking': 0.0},
            'DEF': {'estimated_performance': 0.15, 'estimated_potential': 0.55, 'estimated_regularity': 0.15, 'estimated_goals': 0.05, 'team_ranking': 0.10},
            'MID': {'estimated_performance': 0.10, 'estimated_potential': 0.55, 'estimated_regularity': 0.15, 'estimated_goals': 0.10, 'team_ranking': 0.10},
            'FWD': {'estimated_performance': 0.05, 'estimated_potential': 0.50, 'estimated_regularity': 0.10, 'estimated_goals': 0.25, 'team_ranking': 0.10}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.25},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.35},
            'MID': {'max_proportional_bonus_at_pvs100': 0.5},
            'FWD': {'max_proportional_bonus_at_pvs100': 0.7}
        }
    },
    "Goal Focus": {
        "kpi_weights": {
            'GK': {'estimated_performance': 0.50, 'estimated_potential': 0.30, 'estimated_regularity': 0.20, 'estimated_goals': 0.0, 'team_ranking': 0.0},
            'DEF': {'estimated_performance': 0.20, 'estimated_potential': 0.10, 'estimated_regularity': 0.20, 'estimated_goals': 0.30, 'team_ranking': 0.20},
            'MID': {'estimated_performance': 0.15, 'estimated_potential': 0.10, 'estimated_regularity': 0.15, 'estimated_goals': 0.40, 'team_ranking': 0.20},
            'FWD': {'estimated_performance': 0.10, 'estimated_potential': 0.10, 'estimated_regularity': 0.10, 'estimated_goals': 0.60, 'team_ranking': 0.10}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.2},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.3},
            'MID': {'max_proportional_bonus_at_pvs100': 0.4},
            'FWD': {'max_proportional_bonus_at_pvs100': 0.9}
        }
    }
}

# ----- Data helpers -----
def simplify_position(position: str) -> str:
    if pd.isna(position) or str(position).strip() == '':
        return 'UNKNOWN'
    pos = str(position).upper().strip()
    if pos == 'G': return 'GK'
    elif pos in ['D', 'DL', 'DC']: return 'DEF'
    elif pos in ['M', 'MD', 'MO']: return 'MID'
    elif pos == 'A': return 'FWD'
    else: return 'UNKNOWN'

def create_player_id(row) -> str:
    name = str(row.get('Joueur', '')).strip()
    simplified_pos = simplify_position(row.get('Poste', ''))
    club = str(row.get('Club', '')).strip()
    return f"{name}_{simplified_pos}_{club}"

def extract_rating_goals(rating_str) -> Tuple[Optional[float], int]:
    if pd.isna(rating_str) or str(rating_str).strip() == '' or str(rating_str).strip() == '0':
        return None, 0
    val_str = str(rating_str).strip()
    goals = val_str.count('*')
    clean_rating_str = re.sub(r'[()\*]', '', val_str)
    try:
        rating = float(clean_rating_str)
        return rating, goals
    except ValueError:
        return None, 0

def get_gameweek_columns(df_columns: List[str]) -> List[str]:
    gw_cols = [col for col in df_columns if re.fullmatch(r'D\d+', col)]
    gw_cols_sorted = sorted(gw_cols, key=lambda x: int(x[1:]))  # D1, D2, ..., D34
    return gw_cols_sorted

def calculate_historical_kpis(df_hist: pd.DataFrame) -> pd.DataFrame:
    rdf = df_hist.copy()
    all_gws = get_gameweek_columns(df_hist.columns)
    rdf[['estimated_performance', 'estimated_potential', 'estimated_regularity', 'estimated_goals']] = 0.0
    for idx, row in rdf.iterrows():
        ratings, goals = [], 0
        games_played = 0
        for gw_col in all_gws:
            rating, game_goals = extract_rating_goals(row.get(gw_col))
            if rating is not None:
                ratings.append(rating)
                goals += game_goals
                games_played += 1
        if ratings:
            rdf.at[idx, 'estimated_performance'] = np.mean(ratings)
            rdf.at[idx, 'estimated_potential'] = np.mean(sorted(ratings, reverse=True)[:5]) if len(ratings) >= 5 else np.mean(ratings)
            rdf.at[idx, 'estimated_regularity'] = (games_played / len(all_gws) * 100) if all_gws else 0
            rdf.at[idx, 'estimated_goals'] = goals
    return rdf

def normalize_kpis(df_all: pd.DataFrame, max_perf, max_pot, max_reg, max_goals) -> pd.DataFrame:
    rdf = df_all.copy()
    rdf['norm_estimated_performance'] = np.clip(rdf['estimated_performance'] / max_perf * 100 if max_perf>0 else 0, 0, 100)
    rdf['norm_estimated_potential']   = np.clip(rdf['estimated_potential'] / max_pot * 100 if max_pot>0 else 0, 0, 100)
    rdf['norm_estimated_regularity']  = np.clip(rdf['estimated_regularity'] / max_reg * 100 if max_reg>0 else 0, 0, 100)
    rdf['norm_estimated_goals']       = np.clip(rdf['estimated_goals'] / max_goals * 100 if max_goals>0 else 0, 0, 100)
    rdf['norm_team_ranking']          = rdf['team_ranking']
    return rdf

def calculate_pvs(df: pd.DataFrame, weights: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rdf = df.copy()
    rdf['pvs'] = 0.0
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        mask = rdf['simplified_position'] == pos
        if not mask.any():
            continue
        w = weights[pos]
        total_weight = sum(w.values())
        if total_weight == 0:
            total_weight = 1.0  # Avoid division by zero
        pvs_raw = (
            rdf.loc[mask, 'norm_estimated_performance'] * w.get('estimated_performance', 0) +
            rdf.loc[mask, 'norm_estimated_potential'] * w.get('estimated_potential', 0) +
            rdf.loc[mask, 'norm_estimated_regularity'] * w.get('estimated_regularity', 0) +
            rdf.loc[mask, 'norm_estimated_goals'] * w.get('estimated_goals', 0) +
            rdf.loc[mask, 'norm_team_ranking'] * w.get('team_ranking', 0)
        )
        rdf.loc[mask, 'pvs'] = (pvs_raw / total_weight).clip(0, 100)
    return rdf

def calculate_mrb(df: pd.DataFrame, mrb_params_per_pos: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    rdf = df.copy()
    rdf['mrb'] = rdf['Cote']
    for pos, params in mrb_params_per_pos.items():
        mask = rdf['simplified_position'] == pos
        if not mask.any(): continue
        max_prop_bonus = params.get('max_proportional_bonus_at_pvs100', 0.5)
        def _calc_mrb(row):
            cote = int(row['Cote']); pvs = float(row['pvs'])
            mrb_float = cote * (1 + (pvs/100)*max_prop_bonus)
            return int(round(min(mrb_float, cote*2)))
        rdf.loc[mask, 'mrb'] = rdf.loc[mask].apply(_calc_mrb, axis=1)
    rdf['mrb'] = rdf['mrb'].astype(int)
    safe_mrb = rdf['mrb'].replace(0, 1).astype(float)
    rdf['value_per_cost'] = rdf['pvs'] / safe_mrb
    rdf['value_per_cost'].fillna(0, inplace=True)
    return rdf

def build_gw_strings(row, hist_df):
    if not row.get('is_historical', False):
        return "", ""
    hist_row = hist_df[hist_df['player_id'] == row['player_id']]
    if hist_row.empty:
        return "", ""
    hist_row = hist_row.iloc[0]
    gw_cols = get_gameweek_columns(hist_row.index)
    ratings = []
    goals = []
    for gw in gw_cols:
        r, g = extract_rating_goals(hist_row[gw])
        ratings.append(str(r) if r is not None else "")
        goals.append(str(g))
    return "|".join(ratings), "|".join(goals)

def display_squad_formation(squad_df, formation_key):
    formations = {
        "4-4-2": [("FWD", 2), ("MID", 4), ("DEF", 4), ("GK", 1)],
        "4-3-3": [("FWD", 3), ("MID", 3), ("DEF", 4), ("GK", 1)],
        "3-5-2": [("FWD", 2), ("MID", 5), ("DEF", 3), ("GK", 1)],
        "3-4-3": [("FWD", 3), ("MID", 4), ("DEF", 3), ("GK", 1)],
        "4-5-1": [("FWD", 1), ("MID", 5), ("DEF", 4), ("GK", 1)],
        "5-3-2": [("FWD", 2), ("MID", 3), ("DEF", 5), ("GK", 1)],
        "5-4-1": [("FWD", 1), ("MID", 4), ("DEF", 5), ("GK", 1)]
    }
    st.markdown("### **Visual Formation**")
    squad_df = squad_df.copy()
    squad_df = squad_df[squad_df['is_starter']]
    for pos, n in formations[formation_key]:
        players = squad_df[squad_df['simplified_position']==pos].sort_values('pvs_in_squad', ascending=False).head(n)
        cols = st.columns(n)
        for i, (_, p) in enumerate(players.iterrows()):
            cols[i].markdown(
                f"<div style='text-align:center;'><b>{p['Joueur']}</b><br><span style='font-size:0.85em;'>{p['Club']}</span><br><span style='color:#004080;'>PVS: {p['pvs_in_squad']:.1f}</span></div>",
                unsafe_allow_html=True
            )

def save_dict_to_download_button(data_dict, label, fname):
    bio = io.BytesIO()
    bio.write(json.dumps(data_dict, indent=2).encode('utf-8'))
    bio.seek(0)
    st.download_button(label, data=bio, file_name=fname, mime='application/json')

def load_dict_from_file(uploaded_file):
    if uploaded_file is None:
        return {}
    try:
        content = uploaded_file.read()
        return json.loads(content.decode('utf-8'))
    except Exception as e:
        st.error(f"Could not load file: {e}")
        return {}

class SquadBuilder:
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

    def select_squad(self, df_evaluated_players: pd.DataFrame, formation_key: str, target_squad_size: int) -> Tuple[pd.DataFrame, Dict]:
        eligible_df = df_evaluated_players.drop_duplicates(subset=['player_id']).copy()
        eligible_df['mrb'] = eligible_df['mrb'].astype(int)
        current_squad = []
        def get_current_squad_player_ids_set() -> Set[str]:
            return {p['player_id'] for p in current_squad}
        def get_current_pos_counts_dict() -> Dict[str, int]:
            counts = {pos_key: 0 for pos_key in self.squad_minimums.keys()}
            for p_dict in current_squad:
                counts[p_dict['pos']] = counts.get(p_dict['pos'], 0) + 1
            return counts
        def add_player_to_current_squad_list(player_row_data: pd.Series, is_starter_role: bool) -> bool:
            player_id_to_add = player_row_data['player_id']
            if player_id_to_add in get_current_squad_player_ids_set(): return False
            if player_row_data['simplified_position'] == 'GK':
                if get_current_pos_counts_dict().get('GK', 0) >= 2: return False
            current_squad.append({'player_id': player_id_to_add, 'mrb': int(player_row_data['mrb']),'pvs': float(player_row_data['pvs']), 'pos': player_row_data['simplified_position'],'is_starter': is_starter_role})
            return True
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
        while len(current_squad) < target_squad_size:
            current_counts_ph_a3 = get_current_pos_counts_dict(); most_needed_pos_details = []
            for pos_key, num_starters in self.formations[formation_key].items():
                desired_for_pos = num_starters + 1; deficit = desired_for_pos - current_counts_ph_a3.get(pos_key, 0)
                most_needed_pos_details.append((pos_key, deficit, num_starters))
            most_needed_pos_details.sort(key=lambda x: (x[1], x[2]), reverse=True); player_added_in_a3_pass = False
            for pos_to_fill, deficit_val, _ in most_needed_pos_details:
                if deficit_val <= 0 and len(current_squad) < target_squad_size : pass
                elif deficit_val <=0: continue
                if pos_to_fill == 'GK' and current_counts_ph_a3.get('GK', 0) >= 2: continue
                candidate_series_a3 = all_players_sorted_pvs[(all_players_sorted_pvs['simplified_position'] == pos_to_fill) & (~all_players_sorted_pvs['player_id'].isin(get_current_squad_player_ids_set()))].head(1)
                if not candidate_series_a3.empty:
                    if add_player_to_current_squad_list(candidate_series_a3.iloc[0], False): player_added_in_a3_pass = True; break
            if not player_added_in_a3_pass:
                if len(current_squad) < target_squad_size:
                    candidate_overall_a3 = all_players_sorted_pvs[~all_players_sorted_pvs['player_id'].isin(get_current_squad_player_ids_set())].head(1)
                    if not candidate_overall_a3.empty:
                        if not add_player_to_current_squad_list(candidate_overall_a3.iloc[0], False):
                            all_players_sorted_pvs = all_players_sorted_pvs[all_players_sorted_pvs['player_id'] != candidate_overall_a3.iloc[0]['player_id']]
                            if all_players_sorted_pvs.empty: break
                            continue
                    else: break
                else: break
            if len(current_squad) >= target_squad_size: break
        current_total_mrb = sum(p['mrb'] for p in current_squad)
        max_budget_iterations_b = target_squad_size * 2
        iterations_b_count = 0
        budget_conformance_tolerance = 1
        while current_total_mrb > self.budget + budget_conformance_tolerance and iterations_b_count < max_budget_iterations_b:
            iterations_b_count += 1; made_a_downgrade_in_pass = False; best_downgrade_action = None
            candidates_for_replacement = sorted([p for p in current_squad if not p['is_starter']], key=lambda x: x['mrb'], reverse=True)
            if not candidates_for_replacement and current_total_mrb > self.budget: candidates_for_replacement = sorted(current_squad, key=lambda x: x['pvs'])
            for old_player_dict_b in candidates_for_replacement:
                old_pid_b = old_player_dict_b['player_id']; old_pos_b = old_player_dict_b['pos']; old_mrb_b = old_player_dict_b['mrb']; old_pvs_b = old_player_dict_b['pvs']
                potential_replacements_df = eligible_df[(eligible_df['simplified_position'] == old_pos_b) & (~eligible_df['player_id'].isin(get_current_squad_player_ids_set() - {old_pid_b})) & (eligible_df['mrb'] < old_mrb_b)]
                if not potential_replacements_df.empty:
                    for _, new_player_row_b in potential_replacements_df.iterrows():
                        mrb_saved_b = old_mrb_b - new_player_row_b['mrb']; pvs_change_val_b = new_player_row_b['pvs'] - old_pvs_b
                        current_swap_score_b = mrb_saved_b - (abs(pvs_change_val_b) * 0.5 if pvs_change_val_b < 0 else -pvs_change_val_b * 0.1)
                        if best_downgrade_action is None or current_swap_score_b > best_downgrade_action[4]:
                            best_downgrade_action = (old_pid_b, new_player_row_b['player_id'], mrb_saved_b, pvs_change_val_b, current_swap_score_b, new_player_row_b)
            if best_downgrade_action:
                old_id_exec, new_id_exec, mrb_s_exec, pvs_c_exec, _, new_player_data_exec = best_downgrade_action
                original_starter_status_exec = next((p['is_starter'] for p in current_squad if p['player_id'] == old_id_exec), False)
                current_squad = [p for p in current_squad if p['player_id'] != old_id_exec]
                add_player_to_current_squad_list(new_player_data_exec, original_starter_status_exec)
                current_total_mrb = sum(p['mrb'] for p in current_squad)
                made_a_downgrade_in_pass = True
            if not made_a_downgrade_in_pass:
                if current_total_mrb > self.budget + budget_conformance_tolerance: st.warning(f"Budget Target Not Met: MRB {current_total_mrb} > Budget {self.budget}.")
                break
        final_squad_player_ids = get_current_squad_player_ids_set()
        final_squad_df_base = eligible_df[eligible_df['player_id'].isin(final_squad_player_ids)].copy()
        details_df_final = pd.DataFrame(current_squad)
        details_df_final_renamed = details_df_final.rename(columns={'mrb': 'mrb_actual_cost', 'pvs':'pvs_in_squad', 'is_starter':'is_starter_from_selection'})
        final_squad_df = pd.merge(final_squad_df_base, details_df_final_renamed[['player_id', 'mrb_actual_cost', 'pvs_in_squad', 'is_starter_from_selection']], on='player_id', how='left')
        final_squad_df['mrb_actual_cost'] = final_squad_df['mrb_actual_cost'].fillna(0).astype(int)
        final_squad_df['pvs_in_squad'] = final_squad_df['pvs_in_squad'].fillna(0.0)
        final_starter_ids_definitive = set()
        temp_formation_needs_final = self.formations[formation_key].copy()
        final_squad_df_sorted_for_final_starters = final_squad_df.sort_values(by='pvs_in_squad', ascending=False)
        for _, player_row_final_pass in final_squad_df_sorted_for_final_starters.iterrows():
            pos_final_pass = player_row_final_pass['simplified_position']; player_id_final_pass = player_row_final_pass['player_id']
            if temp_formation_needs_final.get(pos_final_pass, 0) > 0:
                if player_id_final_pass not in final_starter_ids_definitive : final_starter_ids_definitive.add(player_id_final_pass); temp_formation_needs_final[pos_final_pass] -=1
        final_squad_df['is_starter'] = final_squad_df['player_id'].isin(final_starter_ids_definitive)
        if 'is_starter_from_selection' in final_squad_df.columns: final_squad_df.drop(columns=['is_starter_from_selection'], inplace=True, errors='ignore')
        final_total_mrb_actual = final_squad_df['mrb_actual_cost'].sum()
        summary = {
            'total_players': len(final_squad_df),
            'total_cost': int(final_total_mrb_actual),
            'remaining_budget': int(self.budget - final_total_mrb_actual),
            'position_counts': final_squad_df['simplified_position'].value_counts().to_dict(),
            'total_squad_pvs': float(final_squad_df['pvs_in_squad'].sum()),
            'total_starters_pvs': float(final_squad_df[final_squad_df['is_starter']]['pvs_in_squad'].sum())
        }
        return final_squad_df, summary

# ---- MAIN APP ----
def main():
    st.markdown('<h1 class="main-header">🌟 MPG Auction Strategist - New Season Mode</h1>', unsafe_allow_html=True)
    squad_builder = SquadBuilder()
    # --- SIDEBAR: File Inputs and Squad Params ---
    st.sidebar.markdown('<h2 class="section-header" style="margin-top:0;">⚙️ Data Files</h2>', unsafe_allow_html=True)
    hist_file = st.sidebar.file_uploader("Last Season Player Data (CSV/Excel)", type=['csv','xlsx','xls'], key="hist_file")
    new_file = st.sidebar.file_uploader("New Season Players File (CSV/Excel)", type=['csv','xlsx','xls'], key="new_file")
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 👥 Squad Building Parameters")
    formation_key_ui = st.sidebar.selectbox("Preferred Formation", options=list(squad_builder.formations.keys()), index=0)
    target_squad_size_ui = st.sidebar.number_input("Target Squad Size", min_value=sum(squad_builder.squad_minimums.values()), max_value=30, value=DEFAULT_SQUAD_SIZE)
    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🎨 Settings Profile")
    profile_names = list(PREDEFINED_PROFILES.keys())
    if "profile_name" not in st.session_state:
        st.session_state["profile_name"] = "Balanced Value"
    selected_profile_name_ui = st.sidebar.selectbox("Select Profile", options=profile_names, index=profile_names.index(st.session_state["profile_name"]), key="profile_selector")
    if selected_profile_name_ui != st.session_state["profile_name"]:
        st.session_state["profile_name"] = selected_profile_name_ui
    profile_vals = PREDEFINED_PROFILES.get(st.session_state["profile_name"], PREDEFINED_PROFILES["Balanced Value"])
    with st.sidebar.expander("📊 KPI Weights (Click to Customize)", expanded=(st.session_state["profile_name"]=="Custom")):
        weights_ui = {}
        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            st.markdown(f"<h6>{pos}</h6>", unsafe_allow_html=True)
            default_w = PREDEFINED_PROFILES["Balanced Value"]["kpi_weights"][pos]
            current_w = profile_vals["kpi_weights"][pos] if st.session_state["profile_name"]!="Custom" else st.session_state.get("kpi_weights", {}).get(pos, default_w)
            weights_ui[pos] = {
                'estimated_performance': st.slider(f"Performance", 0.0, 1.0, float(current_w.get('estimated_performance', 0.0)), 0.01, key=f"{pos}_wPerf"),
                'estimated_potential': st.slider(f"Potential", 0.0, 1.0, float(current_w.get('estimated_potential', 0.0)), 0.01, key=f"{pos}_wPot"),
                'estimated_regularity': st.slider(f"Regularity", 0.0, 1.0, float(current_w.get('estimated_regularity', 0.0)), 0.01, key=f"{pos}_wReg"),
                'estimated_goals': st.slider(f"Goals", 0.0, 1.0, float(current_w.get('estimated_goals', 0.0)), 0.01, key=f"{pos}_wGoals"),
                'team_ranking': st.slider(f"Team Ranking", 0.0, 1.0, float(current_w.get('team_ranking', 0.0)), 0.01, key=f"{pos}_wTeam"),
            }
        st.session_state["kpi_weights"] = weights_ui if st.session_state["profile_name"]=="Custom" else profile_vals["kpi_weights"]
    with st.sidebar.expander("💰 MRB Parameters (Click to Customize)", expanded=(st.session_state["profile_name"]=="Custom")):
        mrb_params_ui = {}
        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            st.markdown(f"<h6>{pos}</h6>", unsafe_allow_html=True)
            default_mrb = PREDEFINED_PROFILES["Balanced Value"]["mrb_params_per_pos"][pos]
            current_mrb = profile_vals["mrb_params_per_pos"][pos] if st.session_state["profile_name"]!="Custom" else st.session_state.get("mrb_params", {}).get(pos, default_mrb)
            mrb_params_ui[pos] = {'max_proportional_bonus_at_pvs100': st.slider(f"Max Bonus (at PVS 100)", 0.0, 1.0, float(current_mrb.get('max_proportional_bonus_at_pvs100', 0.2)), 0.01, key=f"{pos}_mrb")}
        st.session_state["mrb_params"] = mrb_params_ui if st.session_state["profile_name"]=="Custom" else profile_vals["mrb_params_per_pos"]

    if "zoom_pid" not in st.session_state:
        st.session_state["zoom_pid"] = None

    df_hist, df_new = None, None
    if hist_file:
        try:
            df_hist = pd.read_excel(hist_file) if hist_file.name.endswith(('.xlsx','.xls')) else pd.read_csv(hist_file)
        except Exception as e:
            st.error(f"Could not read historical file: {e}")
    if new_file:
        try:
            df_new = pd.read_excel(new_file) if new_file.name.endswith(('.xlsx','.xls')) else pd.read_csv(new_file)
        except Exception as e:
            st.error(f"Could not read new season file: {e}")

    if df_hist is not None and df_new is not None:
        df_hist['simplified_position'] = df_hist['Poste'].apply(simplify_position)
        df_hist['player_id'] = df_hist.apply(create_player_id, axis=1)
        df_new['simplified_position'] = df_new['Poste'].apply(simplify_position)
        df_new['player_id'] = df_new.apply(create_player_id, axis=1)
        df_hist['Cote'] = pd.to_numeric(df_hist['Cote'], errors='coerce').fillna(1).clip(lower=1).round().astype(int)
        df_new['Cote'] = pd.to_numeric(df_new['Cote'], errors='coerce').fillna(1).clip(lower=1).round().astype(int)
        hist_pids = set(df_hist['player_id'])
        df_new['is_historical'] = df_new['player_id'].isin(hist_pids)
        df_hist_kpis = calculate_historical_kpis(df_hist)
        all_clubs = sorted(df_new['Club'].unique())

        # ---- CLUB TIER UI ----
        with st.expander("🏅 Assign Club Tiers", expanded=False):
            st.write("Assign a tier to each club below:")
            club_tiers = st.session_state.get("club_tiers", {club: "Average" for club in all_clubs})
            col1, col2 = st.columns([1,1])
            with col1:
                save_dict_to_download_button(club_tiers, "💾 Download Club Tiers", "club_tiers.json")
            with col2:
                club_upload = st.file_uploader("⬆️ Load Club Tiers", type=["json"], key="clubtier_upload")
                if club_upload:
                    loaded_tiers = load_dict_from_file(club_upload)
                    if set(loaded_tiers.keys()) == set(all_clubs):
                        club_tiers = loaded_tiers
                        st.success("Club tiers loaded!")
                    else:
                        st.warning("Club list does not match current clubs. Tiers not loaded.")
            cols = st.columns([3, 2, 2, 2, 2])
            for i, club in enumerate(all_clubs):
                tier = cols[i % 5].selectbox(club, CLUB_TIERS_LABELS, index=CLUB_TIERS_LABELS.index(club_tiers.get(club,"Average")), key=f"clubtier_{club}")
                club_tiers[club] = tier
            st.session_state["club_tiers"] = club_tiers

        # ---- Merge all player base info ----
        merged_rows = []
        for idx, row in df_new.iterrows():
            base = row.to_dict()
            club = base['Club']
            base['team_ranking'] = CLUB_TIERS[st.session_state["club_tiers"][club]]
            if base['is_historical']:
                hist_row = df_hist_kpis[df_hist_kpis['player_id']==base['player_id']]
                for col in ['estimated_performance','estimated_potential','estimated_regularity','estimated_goals']:
                    base[col] = float(hist_row.iloc[0][col]) if not hist_row.empty else 0.0
            merged_rows.append(base)
        df_all = pd.DataFrame(merged_rows)

        # ---- NEW PLAYERS UI ----
        with st.expander("🆕 Assign Scores to New Players", expanded=False):
            new_players = df_all[~df_all['is_historical']]
            if "new_player_scores" not in st.session_state:
                st.session_state["new_player_scores"] = {}
            max_perf = df_all[df_all['is_historical']]['estimated_performance'].max() if (df_all['is_historical'].any()) else 1.0
            max_pot  = df_all[df_all['is_historical']]['estimated_potential'].max() if (df_all['is_historical'].any()) else 1.0
            max_reg  = df_all[df_all['is_historical']]['estimated_regularity'].max() if (df_all['is_historical'].any()) else 1.0
            max_goals= df_all[df_all['is_historical']]['estimated_goals'].max() if (df_all['is_historical'].any()) else 1.0
            col1, col2 = st.columns([1,1])
            with col1:
                save_dict_to_download_button(st.session_state["new_player_scores"], "💾 Download New Player Scores", "new_player_scores.json")
            with col2:
                np_upload = st.file_uploader("⬆️ Load New Player Scores", type=["json"], key="npscore_upload")
                if np_upload:
                    loaded_scores = load_dict_from_file(np_upload)
                    st.session_state["new_player_scores"].update(loaded_scores)
                    st.success("New player scores loaded!")
            if not new_players.empty:
                st.write("Rate new players (0, 25, 50, 75, 100% of max historical for each KPI):")
                grid_cols = st.columns([2,1,1,1,1,1])
                grid_cols[0].markdown("**Player**")
                grid_cols[1].markdown("**Perf**")
                grid_cols[2].markdown("**Pot**")
                grid_cols[3].markdown("**Reg**")
                grid_cols[4].markdown("**Goals**")
                for i, nprow in new_players.iterrows():
                    pid = nprow['player_id']
                    if pid not in st.session_state["new_player_scores"]:
                        st.session_state["new_player_scores"][pid] = {
                            "estimated_performance": 0,
                            "estimated_potential": 0,
                            "estimated_regularity": 0,
                            "estimated_goals": 0,
                        }
                    cols = st.columns([2,1,1,1,1])
                    cols[0].markdown(f"{nprow['Joueur']} ({nprow['simplified_position']} - {nprow['Club']})")
                    for ci, (kpi, maxval, label) in enumerate([
                        ("estimated_performance", max_perf, "Perf"),
                        ("estimated_potential", max_pot, "Pot"),
                        ("estimated_regularity", max_reg, "Reg"),
                        ("estimated_goals", max_goals, "Goals"),
                    ], 1):
                        sel = cols[ci].selectbox(
                            "", NEW_PLAYER_SCORE_OPTIONS,
                            index=NEW_PLAYER_SCORE_OPTIONS.index(st.session_state["new_player_scores"][pid][kpi]),
                            key=f"{pid}_{kpi}")
                        st.session_state["new_player_scores"][pid][kpi] = sel
                    for kpi, maxval in [
                        ("estimated_performance", max_perf),
                        ("estimated_potential", max_pot),
                        ("estimated_regularity", max_reg),
                        ("estimated_goals", max_goals)
                    ]:
                        score_pct = st.session_state["new_player_scores"][pid][kpi]
                        df_all.loc[df_all['player_id']==pid, kpi] = (score_pct/100) * maxval
            else:
                st.info("No new players to rate.")

        df_all = normalize_kpis(df_all, max_perf, max_pot, max_reg, max_goals)
        df_all = calculate_pvs(df_all, st.session_state["kpi_weights"])
        df_all = calculate_mrb(df_all, st.session_state["mrb_params"])
        df_all['Ratings per GW'], df_all['Goals per GW'] = zip(
            *df_all.apply(lambda row: build_gw_strings(row, df_hist), axis=1)
        )

        st.markdown('<h2 class="section-header">🏆 Suggested Squad</h2>', unsafe_allow_html=True)
        squad_df, squad_summary = squad_builder.select_squad(df_all, formation_key_ui, target_squad_size_ui)
        if not squad_df.empty:
            squad_disp = squad_df.copy()
            squad_disp = squad_disp.rename(columns={
                "Joueur": "Player", "simplified_position":"Pos", "pvs_in_squad":"PVS", "Cote":"Cote",
                "mrb_actual_cost":"Bid", "estimated_performance":"Perf","estimated_potential":"Pot",
                "estimated_regularity":"Reg", "estimated_goals":"Goals", "team_ranking":"TeamRank"
            })
            squad_disp['Starter'] = squad_disp['is_starter'].map({True:"Yes",False:"No"})
            squad_disp['Ratings per GW'], squad_disp['Goals per GW'] = zip(
                *squad_disp.apply(lambda row: build_gw_strings(row, df_hist), axis=1)
            )
            squad_disp_show = squad_disp[['Player','Club','Pos','PVS','Bid','Perf','Pot','Reg','Goals','TeamRank','Starter','Ratings per GW','Goals per GW']]
            st.dataframe(squad_disp_show, use_container_width=True, hide_index=True)
            display_squad_formation(squad_df, formation_key_ui)
            st.markdown('<h2 class="section-header">📈 Squad Summary</h2>', unsafe_allow_html=True)
            st.metric("Budget Spent", f"€ {squad_summary.get('total_cost',0):.0f}", help=f"Remaining: € {squad_summary.get('remaining_budget',0):.0f}")
            st.metric("Squad Size", f"{squad_summary.get('total_players',0)} (Target: {target_squad_size_ui})")
            st.metric("Total Squad PVS", f"{squad_summary.get('total_squad_pvs',0):.2f}")
            st.metric("Starters PVS", f"{squad_summary.get('total_starters_pvs',0):.2f}")
            st.info(f"**Formation:** {formation_key_ui}")
            for pos in ['GK','DEF','MID','FWD']:
                c = squad_summary.get('position_counts',{}).get(pos,0)
                minr = squad_builder.squad_minimums.get(pos,0)
                st.markdown(f"• **{pos}:** {c} (Min: {minr})")
            st.download_button(label="📥 Download Squad (CSV)", data=squad_disp_show.to_csv(index=False).encode('utf-8'), file_name="mpg_suggested_squad.csv", mime="text/csv")
        else:
            st.warning("Could not build a valid squad. Check your data and settings.")

        st.markdown('<h2 class="section-header">📋 Full Player Database</h2>', unsafe_allow_html=True)
        disp_df = df_all.rename(columns={
            "Joueur":"Player", "simplified_position":"Pos", "pvs":"PVS", "Cote":"Cote",
            "mrb":"Suggested Bid", "estimated_performance":"Perf","estimated_potential":"Pot",
            "estimated_regularity":"Reg", "estimated_goals":"Goals", "team_ranking":"TeamRank"
        })
        disp_df_show = disp_df[['Player','Club','Pos','PVS','Suggested Bid','Perf','Pot','Reg','Goals','TeamRank','Ratings per GW','Goals per GW']]
        st.dataframe(disp_df_show, use_container_width=True, hide_index=True)
        st.download_button(label="📥 Download Player Database (CSV)", data=disp_df_show.to_csv(index=False).encode('utf-8'), file_name="mpg_full_player_database.csv", mime="text/csv")

    else:
        st.info("Upload BOTH last season and new season player files to start. Example columns: Joueur, Poste, Club, Cote, D1..D34")
        example_hist = pd.DataFrame({'Joueur':['PlayerA','PlayerB'], 'Poste':['A','M'], 'Club':['Club X','Club Y'], 'Cote':[45,30], 'D34':['7.5*','6.5'], 'D33':['(6.0)**','0'], 'D32':['','5.5*']})
        example_new = pd.DataFrame({'Joueur':['PlayerA','PlayerB','PlayerC'], 'Poste':['A','M','D'], 'Club':['Club X','Club Y','Club Z'], 'Cote':[45,30,10]})
        st.markdown("**Example Last Season Data:**")
        st.dataframe(example_hist, use_container_width=True, hide_index=True)
        st.markdown("**Example New Season Data:**")
        st.dataframe(example_new, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
