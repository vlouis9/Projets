import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import json
import matplotlib.pyplot as plt
import seaborn as sns
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
    html[data-theme="light"] {
        --primary: #2563eb;
        --secondary: #10b981;
        --accent: #8b5cf6;
        --background: #f8fafc;
        --card: #ffffff;
        --text: #0f172a;
        --border: #e2e8f0;
    }

    html[data-theme="dark"] {
        --primary: #3b82f6;
        --secondary: #10b981;
        --accent: #a78bfa;
        --background: #0f172a;
        --card: #1e293b;
        --text: #f1f5f9;
        --border: #334155;
    }

    body {
        background-color: var(--background);
        color: var(--text);
        font-family: 'Inter', system-ui, sans-serif;
    }

    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        text-align: center;
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--secondary);
        background: linear-gradient(90deg, var(--primary), var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary);
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 4px solid var(--secondary);
    }

    .stButton>button {
        background: linear-gradient(135deg, var(--primary), var(--accent));
        color: white !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.3);
    }

    .card {
        background-color: var(--card);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border: 1px solid var(--border);
    }

    .metric-card {
        background: var(--card);
        text-align: center;
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid var(--primary);
    }

    .position-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text);
    }

    .GK-tag { background: linear-gradient(135deg, #dbeafe, #93c5fd); }
    .DEF-tag { background: linear-gradient(135deg, #dcfce7, #86efac); }
    .MID-tag { background: linear-gradient(135deg, #fef3c7, #fcd34d); }
    .FWD-tag { background: linear-gradient(135deg, #fee2e2, #fca5a5); }

    .starter-badge {
        background-color: var(--secondary);
        color: white;
        border-radius: 4px;
        padding: 0.15rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
    }

    .progress-bar {
        height: 8px;
        border-radius: 4px;
        background-color: var(--border);
        margin-top: 0.5rem;
        overflow: hidden;
    }

    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--primary), var(--accent));
        border-radius: 4px;
    }

    .player-card {
        width: 120px;
        padding: 1rem;
        margin: 0 0.5rem;
        border-radius: 8px;
        text-align: center;
        background: var(--card);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid var(--border);
        transition: all 0.3s ease;
    }

    .player-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }

    .club-badge {
        width: 24px;
        height: 24px;
        display: inline-block;
        border-radius: 50%;
        background-color: var(--primary);
        color: white;
        font-size: 0.7rem;
        line-height: 24px;
        margin-right: 0.5rem;
    }

    .dataframe th {background-color: var(--border) !important;}
    .dataframe td {border-bottom: 1px solid var(--border);}
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
            "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
            "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
            "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
            "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
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

# ---- ENHANCED FORMATION VISUALIZATION ----
def display_enhanced_formation(squad_df, formation_key):
    formations = {
        "4-4-2": [("GK", 1), ("DEF", 4), ("MID", 4), ("FWD", 2)],
        "4-3-3": [("GK", 1), ("DEF", 4), ("MID", 3), ("FWD", 3)],
        "3-5-2": [("GK", 1), ("DEF", 3), ("MID", 5), ("FWD", 2)],
        "3-4-3": [("GK", 1), ("DEF", 3), ("MID", 4), ("FWD", 3)],
        "4-5-1": [("GK", 1), ("DEF", 4), ("MID", 5), ("FWD", 1)],
        "5-3-2": [("GK", 1), ("DEF", 5), ("MID", 3), ("FWD", 2)],
        "5-4-1": [("GK", 1), ("DEF", 5), ("MID", 4), ("FWD", 1)]
    }
    
    squad_df = squad_df.copy().sort_values('pvs_in_squad', ascending=False)
    
    for position, count in formations[formation_key]:
        players = squad_df[squad_df['simplified_position'] == position].head(count)
        if players.empty:
            continue
            
        st.markdown(f"<h4>{position} ({count})</h4>", unsafe_allow_html=True)
        cols = st.columns(count)
        
        for i, (_, player) in enumerate(players.iterrows()):
            with cols[i]:
                starter = "🟢 STARTER" if player['is_starter'] else "🔵 BENCH"
                st.markdown(f"""
                <div class="player-card">
                    <div style="font-weight: 700; margin-bottom: 0.5rem;">{player['Joueur']}</div>
                    <div style="display: flex; align-items: center; justify-content: center; margin-bottom: 0.5rem;">
                        <span class="club-badge">{player['Club'][:2]}</span>
                        <span>{player['Club']}</span>
                    </div>
                    <div class="{player['simplified_position']}-tag position-tag" style="margin-bottom: 0.5rem;">
                        {player['simplified_position']}
                    </div>
                    <div style="font-weight: 600; color: var(--primary); margin-bottom: 0.25rem;">
                        PVS: {player['pvs_in_squad']:.1f}
                    </div>
                    <div style="font-size: 0.85em; color: var(--text);">
                        Bid: €{player['mrb_actual_cost']}
                    </div>
                    <div style="margin-top: 0.5rem; font-size: 0.75rem;" class="starter-badge">
                        {starter}
                    </div>
                </div>
                """, unsafe_allow_html=True)

# ---- PLAYER PERFORMANCE VISUALIZATION ----
def is_dark_mode():
    bg_color = st.get_option("theme.backgroundColor")
    return bg_color and bg_color.lower() in ["#0f172a", "#1e293b"]

def set_plot_theme():
    if is_dark_mode():
        plt.style.use('dark_background')
        sns.set_style("darkgrid")
    else:
        plt.style.use('default')
        sns.set_style("whitegrid")

def plot_player_performance(player_row, df_hist):
    if not player_row['is_historical']:
        st.warning("No historical data available for new players")
        return

    hist_row = df_hist[df_hist['player_id'] == player_row['player_id']]
    if hist_row.empty:
        st.warning("Historical data not found for this player")
        return

    hist_row = hist_row.iloc[0]
    gw_cols = get_gameweek_columns(hist_row.index)

    ratings, goals, gameweeks = [], [], []

    for gw in gw_cols:
        r, g = extract_rating_goals(hist_row[gw])
        if r is not None:
            ratings.append(r)
            goals.append(g)
            gameweeks.append(int(gw[1:]))

    if not ratings:
        st.warning("No performance data available for this player")
        return

    set_plot_theme()  # ✅ Set correct theme before plotting

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)

    sns.lineplot(x=gameweeks, y=ratings, ax=ax1, color="#3b82f6", marker='o', linewidth=2.5)
    ax1.set_title(f"{player_row['Joueur']} - Ratings per Gameweek", fontsize=16)
    ax1.set_ylabel("Rating")
    ax1.set_ylim(0, 10)
    ax1.grid(True, linestyle='--', alpha=0.7)

    sns.barplot(x=gameweeks, y=goals, ax=ax2, color="#a78bfa", edgecolor="#3b82f6")
    ax2.set_title(f"{player_row['Joueur']} - Goals per Gameweek", fontsize=16)
    ax2.set_xlabel("Gameweek")
    ax2.set_ylabel("Goals")
    ax2.grid(True, linestyle='--', alpha=0.3)

    for i, v in enumerate(goals):
        if v > 0:
            ax2.text(i, v + 0.1, str(v), ha='center')

    plt.tight_layout()
    st.pyplot(fig)

# ---- MAIN APP ----
def main():
    st.markdown('<h1 class="main-header">🌟 MPG Auction Strategist - New Season Mode</h1>', unsafe_allow_html=True)
    
    # Initialize session state variables
    if 'club_tiers' not in st.session_state:
        st.session_state.club_tiers = {}
    if 'new_player_scores' not in st.session_state:
        st.session_state.new_player_scores = {}
    if 'profile_name' not in st.session_state:
        st.session_state.profile_name = "Balanced Value"
    if 'kpi_weights' not in st.session_state:
        st.session_state.kpi_weights = PREDEFINED_PROFILES["Balanced Value"]["kpi_weights"]
    if 'mrb_params' not in st.session_state:
        st.session_state.mrb_params = PREDEFINED_PROFILES["Balanced Value"]["mrb_params_per_pos"]
    
    squad_builder = SquadBuilder()
    
    # --- SIDEBAR: File Inputs and Settings ---
    with st.sidebar:
        with st.expander("⚙️ Data Files"):
            hist_file = st.file_uploader("Last Season Player Data (CSV/Excel)", type=['csv','xlsx','xls'], key="hist_file")
            new_file = st.file_uploader("New Season Players File (CSV/Excel)", type=['csv','xlsx','xls'], key="new_file")
        st.markdown("---")

    # Use Streamlit's native tabs
    tab1, tab2, tab3, tab4 = st.tabs(["🏆 Squad Builder", "📋 Player Database", "🏅 Club Tiers", "🆕 New Players scores"])

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

    # Initialize club tiers and new player scores if files are uploaded
    if df_hist is not None and df_new is not None:
        # Initialize club tiers
        all_clubs = sorted(df_new['Club'].unique())
        if not st.session_state.club_tiers:
            st.session_state.club_tiers = {club: "Average" for club in all_clubs}
        else:
            # Add any new clubs that might be missing
            for club in all_clubs:
                if club not in st.session_state.club_tiers:
                    st.session_state.club_tiers[club] = "Average"
            
        df_hist['simplified_position'] = df_hist['Poste'].apply(simplify_position)
        df_hist['player_id'] = df_hist.apply(create_player_id, axis=1)
        df_new['simplified_position'] = df_new['Poste'].apply(simplify_position)
        df_new['player_id'] = df_new.apply(create_player_id, axis=1)
        df_hist['Cote'] = df_hist['Cote'].astype(str).str.strip().replace('NaN', '')
        df_hist['Cote'] = pd.to_numeric(df_hist['Cote'], errors='coerce').fillna(1).clip(lower=1).round().astype(int)
        df_new['Cote'] = pd.to_numeric(df_new['Cote'], errors='coerce').fillna(1).clip(lower=1).round().astype(int)
        hist_pids = set(df_hist['player_id'])
        df_new['is_historical'] = df_new['player_id'].isin(hist_pids)
        df_hist_kpis = calculate_historical_kpis(df_hist)
        
        # Initialize new player scores
        new_players = df_new[~df_new['is_historical']]
        if not st.session_state.new_player_scores:
            st.session_state.new_player_scores = {}
        for idx, row in new_players.iterrows():
            pid = row['player_id']
            if pid not in st.session_state.new_player_scores:
                st.session_state.new_player_scores[pid] = {
                    "estimated_performance": 0,
                    "estimated_potential": 0,
                    "estimated_regularity": 0,
                    "estimated_goals": 0,
                }
        
        # Club Tiers Configuration
        with tab3:
            st.markdown("## 🏅 Assign Club Tiers")
            with st.expander("⚙️ Club Files"):
                col1, col2 = st.columns([1,1])
                with col1:
                    save_dict_to_download_button(st.session_state.club_tiers, "💾 Download Club Tiers", "club_tiers.json")
                with col2:
                    club_upload = st.file_uploader("⬆️ Load Club Tiers", type=["json"], key="clubtier_upload")
                    if club_upload:
                        loaded_tiers = load_dict_from_file(club_upload)
                        if set(loaded_tiers.keys()) == set(all_clubs):
                            st.session_state.club_tiers = loaded_tiers
                            st.success("Club tiers loaded!")
                        else:
                            st.warning("Club list does not match current clubs. Tiers not loaded.")
            
            st.markdown("---")
            
            cols = st.columns(5)
            club_cols = [cols[i % 5] for i in range(len(all_clubs))]
            for i, club in enumerate(all_clubs):
                tier = club_cols[i].selectbox(
                    club, 
                    CLUB_TIERS_LABELS, 
                    index=CLUB_TIERS_LABELS.index(st.session_state.club_tiers.get(club,"Average")), 
                    key=f"clubtier_{club}"
                )
                st.session_state.club_tiers[club] = tier
        
        # New Player Ratings
        with tab4:
            st.markdown("## 🆕 Assign Scores to New Players")
            with st.expander("⚙️ New Players Files"):
                col1, col2 = st.columns([1,1])
                with col1:
                    save_dict_to_download_button(st.session_state.new_player_scores, "💾 Download New Player Scores", "new_player_scores.json")
                with col2:
                    np_upload = st.file_uploader("⬆️ Load New Player Scores", type=["json"], key="npscore_upload")
                    if np_upload:
                        loaded_scores = load_dict_from_file(np_upload)
                        st.session_state.new_player_scores.update(loaded_scores)
                        st.success("New player scores loaded!")
           
            st.markdown("---")
            
            if not new_players.empty:
                st.write("Rate new players (0, 25, 50, 75, 100% of max historical for each KPI):")
                new_players_by_club = new_players.groupby('Club')
                for club_name, players_df in new_players_by_club:
                    with st.expander(f"**{club_name}** - {len(players_df)} player(s)", expanded=False):
                        for _, nprow in players_df.iterrows():
                            pid = nprow['player_id']
                            st.markdown(f"**{nprow['Joueur']}** ({nprow['Poste']})")
                            cols = st.columns(4)
                            for kpi, maxval, label in [
                                ("estimated_performance", df_hist_kpis['estimated_performance'].max(), "Performance"),
                                ("estimated_potential", df_hist_kpis['estimated_potential'].max(), "Potential"),
                                ("estimated_regularity", df_hist_kpis['estimated_regularity'].max(), "Regularity"),
                                ("estimated_goals", df_hist_kpis['estimated_goals'].max(), "Goals")
                            ]:
                                with cols[0] if kpi == "estimated_performance" else cols[1] if kpi == "estimated_potential" else cols[2] if kpi == "estimated_regularity" else cols[3]:
                                    sel = st.selectbox(
                                        label, 
                                        NEW_PLAYER_SCORE_OPTIONS,
                                        index=NEW_PLAYER_SCORE_OPTIONS.index(st.session_state.new_player_scores[pid][kpi]),
                                        key=f"{pid}_{kpi}"
                                    )
                                    st.session_state.new_player_scores[pid][kpi] = sel
            else:
                st.info("No new players to rate.")
        
        # Merge all player data
        merged_rows = []
        for idx, row in df_new.iterrows():
            base = row.to_dict()
            club = base['Club']
            base['team_ranking'] = CLUB_TIERS[st.session_state.club_tiers[club]]
            if base['is_historical']:
                hist_row = df_hist_kpis[df_hist_kpis['player_id']==base['player_id']]
                for col in ['estimated_performance','estimated_potential','estimated_regularity','estimated_goals']:
                    base[col] = float(hist_row.iloc[0][col]) if not hist_row.empty else 0.0
            else:
                pid = base['player_id']
                for kpi in ['estimated_performance','estimated_potential','estimated_regularity','estimated_goals']:
                    score_pct = st.session_state.new_player_scores[pid][kpi]
                    if kpi == "estimated_performance": maxval = df_hist_kpis['estimated_performance'].max()
                    elif kpi == "estimated_potential": maxval = df_hist_kpis['estimated_potential'].max()
                    elif kpi == "estimated_regularity": maxval = df_hist_kpis['estimated_regularity'].max()
                    elif kpi == "estimated_goals": maxval = df_hist_kpis['estimated_goals'].max()
                    base[kpi] = (score_pct/100) * maxval
            merged_rows.append(base)
        df_all = pd.DataFrame(merged_rows)

        # Normalize and calculate metrics
        max_perf = df_hist_kpis['estimated_performance'].max()
        max_pot  = df_hist_kpis['estimated_potential'].max()
        max_reg  = df_hist_kpis['estimated_regularity'].max()
        max_goals= df_hist_kpis['estimated_goals'].max()
        
        df_all = normalize_kpis(df_all, max_perf, max_pot, max_reg, max_goals)
        df_all = calculate_pvs(df_all, st.session_state.kpi_weights)
        df_all = calculate_mrb(df_all, st.session_state.mrb_params)
        df_all['Ratings per GW'], df_all['Goals per GW'] = zip(
            *df_all.apply(lambda row: build_gw_strings(row, df_hist), axis=1)
        )
        
        # Tab 1: Squad Builder
        with tab1:
            st.markdown("#### 👥 Squad Building Parameters")
            
            col1, col2 = st.columns([2,8])
            with col1:    
                formation_key_ui = st.selectbox("Preferred Formation", options=list(squad_builder.formations.keys()), index=0)
                target_squad_size_ui = st.number_input("Target Squad Size", min_value=sum(squad_builder.squad_minimums.values()), max_value=30, value=DEFAULT_SQUAD_SIZE)
            
            with col2:
                profile_names = list(PREDEFINED_PROFILES.keys())
                selected_profile_name_ui = st.selectbox("Select Profile", options=profile_names, index=profile_names.index(st.session_state.profile_name), key="profile_selector")
                if selected_profile_name_ui != st.session_state.profile_name:
                    st.session_state.profile_name = selected_profile_name_ui
                profile_vals = PREDEFINED_PROFILES.get(st.session_state.profile_name, PREDEFINED_PROFILES["Balanced Value"])

                if selected_profile_name_ui=="Custom":
                    subcol1, subcol2 = st.columns([4,4])
                    with subcol1:
                        with st.expander("📊 KPI Weights (Click to Customize)", expanded=(st.session_state.profile_name=="Custom")):
                            weights_ui = {}
                            for pos in ['GK', 'DEF', 'MID', 'FWD']:
                                with st.expander(f"**{pos}**"):
                                    default_w = PREDEFINED_PROFILES["Balanced Value"]["kpi_weights"][pos]
                                    current_w = profile_vals["kpi_weights"][pos] if st.session_state.profile_name!="Custom" else st.session_state.kpi_weights.get(pos, default_w)
                                    weights_ui[pos] = {
                                        'estimated_performance': st.slider(f"Performance", 0.0, 1.0, float(current_w.get('estimated_performance', 0.0)), 0.01, key=f"{pos}_wPerf"),
                                        'estimated_potential': st.slider(f"Potential", 0.0, 1.0, float(current_w.get('estimated_potential', 0.0)), 0.01, key=f"{pos}_wPot"),
                                        'estimated_regularity': st.slider(f"Regularity", 0.0, 1.0, float(current_w.get('estimated_regularity', 0.0)), 0.01, key=f"{pos}_wReg"),
                                        'estimated_goals': st.slider(f"Goals", 0.0, 1.0, float(current_w.get('estimated_goals', 0.0)), 0.01, key=f"{pos}_wGoals"),
                                        'team_ranking': st.slider(f"Team Ranking", 0.0, 1.0, float(current_w.get('team_ranking', 0.0)), 0.01, key=f"{pos}_wTeam"),
                                    }
                            if st.session_state.profile_name=="Custom":
                                st.session_state.kpi_weights = weights_ui
                    
                    with subcol2:
                        with st.expander("💰 MRB Parameters (Click to Customize)", expanded=(st.session_state.profile_name=="Custom")):
                            mrb_params_ui = {}
                            for pos in ['GK', 'DEF', 'MID', 'FWD']:
                                with st.expander(f"**{pos}**"):
                                    default_mrb = PREDEFINED_PROFILES["Balanced Value"]["mrb_params_per_pos"][pos]
                                    current_mrb = profile_vals["mrb_params_per_pos"][pos] if st.session_state.profile_name!="Custom" else st.session_state.mrb_params.get(pos, default_mrb)
                                    mrb_params_ui[pos] = {'max_proportional_bonus_at_pvs100': st.slider(f"Max Bonus (at PVS 100)", 0.0, 1.0, float(current_mrb.get('max_proportional_bonus_at_pvs100', 0.2)), 0.01, key=f"{pos}_mrb")}
                            if st.session_state.profile_name=="Custom":
                                st.session_state.mrb_params = mrb_params_ui
            
            st.markdown("---")
            st.markdown('<h2 class="section-header">🏆 Suggested Squad</h2>', unsafe_allow_html=True)
            squad_df, squad_summary = squad_builder.select_squad(df_all, formation_key_ui, target_squad_size_ui)
            
            if not squad_df.empty:
                # Budget Progress Bar
                budget_spent = squad_summary.get('total_cost', 0)
                budget_remaining = squad_summary.get('remaining_budget', 0)
                progress_percent = min(100, (budget_spent / squad_builder.budget) * 100)
                
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.metric("Budget Spent", f"€ {budget_spent:,}")
                with col2:
                    st.caption(f"Remaining: € {budget_remaining:,}")
                    st.markdown(f"""
                    <div class="progress-bar">
                        <div class="progress-fill" style="width: {progress_percent}%"></div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Squad Summary Cards
                cols = st.columns(4)
                with cols[0]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.25rem; font-weight: 700;">{squad_summary.get('total_players',0)}</div>
                        <div>Players</div>
                    </div>
                    """, unsafe_allow_html=True)
                with cols[1]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.25rem; font-weight: 700;">{squad_summary.get('total_squad_pvs',0):.0f}</div>
                        <div>Total PVS</div>
                    </div>
                    """, unsafe_allow_html=True)
                with cols[2]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.25rem; font-weight: 700;">{squad_summary.get('total_starters_pvs',0):.0f}</div>
                        <div>Starters PVS</div>
                    </div>
                    """, unsafe_allow_html=True)
                with cols[3]:
                    st.markdown(f"""
                    <div class="metric-card">
                        <div style="font-size: 1.25rem; font-weight: 700;">{formation_key_ui}</div>
                        <div>Formation</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Enhanced Formation Visualization
                st.markdown("### **Team Formation**")
                display_enhanced_formation(squad_df, formation_key_ui)
                
                # Position Distribution
                st.markdown("### **Position Distribution**")
                pos_counts = squad_summary.get('position_counts', {})
                min_counts = squad_builder.squad_minimums
                
                for pos in ['GK', 'DEF', 'MID', 'FWD']:
                    count = pos_counts.get(pos, 0)
                    min_req = min_counts.get(pos, 0)
                    status = "✅" if count >= min_req else "⚠️"
                    st.progress(
                        min(1.0, count / (min_req * 1.5)), 
                        text=f"{pos}: {count} players {status} (Min: {min_req})"
                    )
                
                # Player Table
                st.markdown("### **Squad Details**")
                squad_disp_show = squad_df[['Joueur','Club','simplified_position','pvs_in_squad','mrb_actual_cost','is_starter']]
                squad_disp_show = squad_disp_show.rename(columns={
                    "Joueur": "Player", 
                    "simplified_position": "Position",
                    "pvs_in_squad": "PVS",
                    "mrb_actual_cost": "Bid",
                    "is_starter": "Starter"
                })
                
                # Format table with badges and colors
                def format_row(row):
                    position = row['Position']
                    starter = "🟢" if row['Starter'] else "🔵"
                    badge = f"<span class='{position}-tag position-tag'>{position}</span>"
                    return [
                        row['Player'],
                        row['Club'],
                        badge,
                        f"{row['PVS']:.1f}",
                        f"€{row['Bid']}",
                        starter
                    ]
                
                formatted_data = [format_row(row) for _, row in squad_disp_show.iterrows()]
                st.write(pd.DataFrame(
                    formatted_data,
                    columns=["Player", "Club", "Position", "PVS", "Bid", "Starter"]
                ).to_html(escape=False, index=False), unsafe_allow_html=True)
                
                st.download_button(
                    label="📥 Download Squad (CSV)", 
                    data=squad_disp_show.to_csv(index=False).encode('utf-8'), 
                    file_name="mpg_suggested_squad.csv", 
                    mime="text/csv"
                )
                
            else:
                st.warning("Could not build a valid squad. Check your data and settings.")
        
        # Tab 2: Player Database
        with tab2:
            st.markdown('<h2 class="section-header">📋 Full Player Database</h2>', unsafe_allow_html=True)
            disp_df = df_all.rename(columns={
                "Joueur": "Player", 
                "simplified_position": "Position",
                "pvs": "PVS", 
                "Cote": "Base Price",
                "mrb": "Suggested Bid", 
                "estimated_performance": "Performance",
                "estimated_potential": "Potential",
                "estimated_regularity": "Regularity",
                "estimated_goals": "Goals", 
                "team_ranking": "Team Rank",
                "is_historical": "Historical"
            })
            disp_df_show = disp_df[[
                'Player', 'Club', 'Position', 'PVS', 'Base Price', 'Suggested Bid', 
                'Performance', 'Potential', 'Regularity', 'Goals', 'Team Rank', 'Historical'
            ]]
            
            # Format position tags
            def format_pos(pos):
                return f"<span class='{pos}-tag position-tag'>{pos}</span>"
            
            disp_df_show['Position'] = disp_df_show['Position'].str.replace(r'<.*?>', '', regex=True)
            
            # Display the player database
            st.dataframe(disp_df_show, use_container_width=True, hide_index=True)
            
            # Player selection for performance visualization
            with st.expander("🔍 Player Performance Analysis", expanded=False):
                st.markdown("Filter and pick a player to visualize ratings & goals per match.")
            
                search_text = st.text_input("Search player name or club")
            
                # Only historical players with real performance
                filtered_df = disp_df[(disp_df['Historical'] == True) & (disp_df['Performance'] > 0)]
            
                if search_text:
                    filtered_df = filtered_df[
                        filtered_df['Player'].str.contains(search_text, case=False, na=False) |
                        filtered_df['Club'].str.contains(search_text, case=False, na=False)
                    ]
            
                if not filtered_df.empty:
                    player_options = filtered_df[['Player', 'Club']].apply(lambda x: f"{x['Player']} ({x['Club']})", axis=1).tolist()
                    selected_player = st.selectbox("Select player", options=player_options)
            
                    if selected_player is not None and isinstance(selected_player, str):
                        selected_player_name = selected_player.split(" (")[0]
                        selected_row = df_all[df_all['Joueur'] == selected_player_name]
                        if not selected_row.empty:
                            plot_player_performance(selected_row.iloc[0], df_hist)
                        else:
                            st.warning("Selected player not found in database.")
                else:
                    st.info("No players found matching the search.")
            
            
            st.download_button(
                label="📥 Download Player Database (CSV)", 
                data=disp_df_show.to_csv(index=False).encode('utf-8'), 
                file_name="mpg_full_player_database.csv", 
                mime="text/csv"
            )
    
    else:
        st.info("Upload BOTH last season and new season player files to start. Example columns: Joueur, Poste, Club, Cote, D1..D34")

if __name__ == "__main__":
    main()
