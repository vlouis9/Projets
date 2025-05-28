# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional

# Page configuration
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

# Constants and Predefined Profiles
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
            'GK': {'max_proportional_bonus_at_pvs100': 0.5},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.6},
            'MID': {'max_proportional_bonus_at_pvs100': 0.8},
            'FWD': {'max_proportional_bonus_at_pvs100': 1.0}
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
            'GK': {'max_proportional_bonus_at_pvs100': 0.4},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.3},
            'MID': {'max_proportional_bonus_at_pvs100': 0.6},
            'FWD': {'max_proportional_bonus_at_pvs100': 0.8}
        }
    },
    "Focus on Season Consistency": {
        "n_recent_games": 7,
        "min_recent_games_played_filter": 2,
        "kpi_weights": {
            'GK': {'recent_avg': 0.15, 'season_avg': 0.4, 'calc_regularity': 0.25, 'recent_goals': 0.0, 'season_goals': 0.0},
            'DEF': {'recent_avg': 0.1, 'season_avg': 0.4, 'calc_regularity': 0.25, 'recent_goals': 0.0, 'season_goals': 0.0},
            'MID': {'recent_avg': 0.1, 'season_avg': 0.3, 'calc_regularity': 0.2, 'recent_goals': 0.05, 'season_goals': 0.15},
            'FWD': {'recent_avg': 0.1, 'season_avg': 0.3, 'calc_regularity': 0.15, 'recent_goals': 0.1, 'season_goals': 0.2}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.2},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.3},
            'MID': {'max_proportional_bonus_at_pvs100': 0.5},
            'FWD': {'max_proportional_bonus_at_pvs100': 0.7}
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

    def create_player_id(self, row) -> str:
        name = str(row.get('Joueur', '')).strip()
        simplified_pos = self.simplify_position(row.get('Poste', ''))
        club = str(row.get('Club', '')).strip()
        return f"{name}_{simplified_pos}_{club}"

    def extract_rating_goals_starter(self, rating_str) -> Tuple[Optional[float], int, bool, bool]:
        """Returns rating, goals, played_this_gw, is_starter"""
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

    def get_gameweek_columns(self, df_columns: List[str]) -> List[str]:
        gw_cols_data = [{'name': col, 'number': int(match.group(1))} for col in df_columns if (match := re.fullmatch(r'D(\d+)', col))]
        return [col['name'] for col in sorted(gw_cols_data, key=lambda x: x['number'])]

    def calculate_kpis(self, df: pd.DataFrame, n_recent: int) -> pd.DataFrame:
        rdf = df.copy()
        all_gws = self.get_gameweek_columns(df.columns)
        rdf[['recent_avg_rating', 'season_avg_rating']] = 0.0
        rdf[['recent_goals', 'season_goals', 'recent_games_played_count',
             'calc_regularity_pct', 'games_started_season', 'total_season_gws_considered']] = 0

        for idx, row in rdf.iterrows():
            s_ratings_p, s_goals_t, s_started, s_played = [], 0, 0, 0
            for gw_col in all_gws:
                r, g, played, starter = self.extract_rating_goals_starter(row.get(gw_col))
                if played and r is not None:
                    s_ratings_p.append(r)
                    s_goals_t += g
                    s_played += 1
                    if starter:
                        s_started += 1
            rdf.at[idx, 'season_avg_rating'] = np.mean(s_ratings_p) if s_ratings_p else 0.0
            rdf.at[idx, 'season_goals'] = s_goals_t
            rdf.at[idx, 'games_started_season'] = s_started
            rdf.at[idx, 'total_season_gws_considered'] = len(all_gws)
            rdf.at[idx, 'calc_regularity_pct'] = (s_started / len(all_gws) * 100) if len(all_gws) > 0 else 0.0

            rec_gws_check = all_gws[-n_recent:]
            rec_ratings_p, rec_goals_s, rec_games_p_window = [], 0, 0
            for gw_col in rec_gws_check:
                r, g, played, _ = self.extract_rating_goals_starter(row.get(gw_col))
                if played and r is not None:
                    rec_ratings_p.append(r)
                    rec_goals_s += g
                    rec_games_p_window += 1
            rdf.at[idx, 'recent_avg_rating'] = np.mean(rec_ratings_p) if rec_ratings_p else 0.0
            rdf.at[idx, 'recent_goals'] = rec_goals_s
            rdf.at[idx, 'recent_games_played_count'] = rec_games_p_window

        for col in ['recent_goals', 'season_goals', 'recent_games_played_count', 'games_started_season', 'total_season_gws_considered']:
            rdf[col] = rdf[col].astype(int)
        return rdf

    def normalize_kpis(self, df: pd.DataFrame) -> pd.DataFrame:
        rdf = df.copy()
        rdf['norm_recent_avg'] = np.clip(rdf['recent_avg_rating'] * 10, 0, 100)
        rdf['norm_season_avg'] = np.clip(rdf['season_avg_rating'] * 10, 0, 100)
        rdf['norm_regularity_file'] = pd.to_numeric(rdf['%Titu'], errors='coerce').fillna(0).clip(0, 100)
        rdf['norm_calc_regularity'] = rdf['calc_regularity_pct'].clip(0, 100)
        rdf[['norm_recent_goals', 'norm_season_goals']] = 0.0
        for pos in ['MID', 'FWD']:
            mask = rdf['simplified_position'] == pos
            if mask.any():
                rdf.loc[mask, 'norm_recent_goals'] = np.clip(rdf.loc[mask, 'recent_goals'] * 20, 0, 100)
                max_sg = rdf.loc[mask, 'season_goals'].max()
                rdf.loc[mask, 'norm_season_goals'] = np.clip((rdf.loc[mask, 'season_goals'] / max_sg * 100) if max_sg > 0 else 0, 0, 100)
        return rdf

    def calculate_pvs(self, df: pd.DataFrame, weights: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        rdf = df.copy()
        rdf['pvs'] = 0.0
        for pos, w in weights.items():
            mask = rdf['simplified_position'] == pos
            if not mask.any():
                continue
            pvs_sum = pd.Series(0.0, index=rdf.loc[mask].index)
            pvs_sum += rdf.loc[mask, 'norm_recent_avg'].fillna(0) * w.get('recent_avg', 0)
            pvs_sum += rdf.loc[mask, 'norm_season_avg'].fillna(0) * w.get('season_avg', 0)
            pvs_sum += rdf.loc[mask, 'norm_calc_regularity'].fillna(0) * w.get('calc_regularity', 0)
            if pos in ['MID', 'FWD']:
                pvs_sum += rdf.loc[mask, 'norm_recent_goals'].fillna(0) * w.get('recent_goals', 0)
                pvs_sum += rdf.loc[mask, 'norm_season_goals'].fillna(0) * w.get('season_goals', 0)
            rdf.loc[mask, 'pvs'] = pvs_sum.clip(0, 100)
        return rdf

    def calculate_mrb(self, df: pd.DataFrame, mrb_params_per_pos: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        rdf = df.copy()
        rdf['Cote'] = pd.to_numeric(rdf['Cote'], errors='coerce').fillna(1).clip(lower=1).round().astype(int)
        rdf['mrb'] = rdf['Cote']
        for pos_simplified, params in mrb_params_per_pos.items():
            mask = rdf['simplified_position'] == pos_simplified
            if not mask.any():
                continue
            max_prop_bonus = params.get('max_proportional_bonus_at_pvs100', 0.5)

            def _calc_mrb_player_v3(row):
                cote = int(row['Cote'])
                pvs_player_0_100 = float(row['pvs'])
                pvs_scaled_0_1 = pvs_player_0_100 / 100.0
                pvs_derived_bonus_factor = pvs_scaled_0_1 * max_prop_bonus
                mrb_float = cote * (1 + pvs_derived_bonus_factor)
                mrb_capped_at_2x_cote = min(mrb_float, float(cote * 2))
                final_mrb = max(float(cote), mrb_capped_at_2x_cote)
                return int(round(final_mrb))

            rdf.loc[mask, 'mrb'] = rdf.loc[mask].apply(_calc_mrb_player_v3, axis=1)
        rdf['mrb'] = rdf['mrb'].astype(int)
        safe_mrb = rdf['mrb'].replace(0, np.nan).astype(float)
        rdf['value_per_cost'] = rdf['pvs'] / safe_mrb
        rdf['value_per_cost'].fillna(0, inplace=True)
        return rdf

    def select_squad(self, df: pd.DataFrame, formation_key: str, target_squad_size: int,
                     min_recent_games_played: int) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        eligible_df_initial = df.copy()
        if min_recent_games_played > 0:
            eligible_df_initial = eligible_df_initial[eligible_df_initial['recent_games_played_count'] >= min_recent_games_played]
        #if 'Indispo ?' in eligible_df_initial.columns:
            #eligible_df_initial = eligible_df_initial[~eligible_df_initial['Indispo ?']]
        if eligible_df_initial.empty:
            return pd.DataFrame(), {}
        eligible_df = eligible_df_initial.drop_duplicates(subset=['player_id']).copy()
        eligible_df['mrb'] = eligible_df['mrb'].astype(int)

        selected_details = []
        current_budget = self.budget
        current_pos_counts = {pos: 0 for pos in ['GK', 'DEF', 'MID', 'FWD']}

        def add_player_to_squad(player_row, is_starter_role):
            nonlocal current_budget, selected_details, current_pos_counts
            if player_row['player_id'] in [p['player_id'] for p in selected_details]:
                return False
            selected_details.append({
                'player_id': player_row['player_id'], 'is_starter': is_starter_role,
                'mrb_cost': player_row['mrb'], 'pvs': player_row['pvs'],
                'position': player_row['simplified_position']
            })
            current_budget -= player_row['mrb']
            current_pos_counts[player_row['simplified_position']] += 1
            return True

        starters_needed = self.formations[formation_key].copy()
        sorted_for_starters = eligible_df.sort_values(by='pvs', ascending=False)
        for _, player_row in sorted_for_starters.iterrows():
            pos = player_row['simplified_position']
            if starters_needed.get(pos, 0) > 0 and player_row['mrb'] <= current_budget:
                if add_player_to_squad(player_row, True):
                    starters_needed[pos] -= 1

        for pos, overall_min in self.squad_minimums.items():
            needed = max(0, overall_min - current_pos_counts[pos])
            if needed == 0:
                continue
            candidates = eligible_df[
                (eligible_df['simplified_position'] == pos) &
                (~eligible_df['player_id'].isin([p['player_id'] for p in selected_details]))
            ].sort_values(by='pvs', ascending=False)
            added_count = 0
            for _, player_row in candidates.iterrows():
                if added_count >= needed:
                    break
                if player_row['mrb'] <= current_budget:
                    if add_player_to_squad(player_row, False):
                        added_count += 1

        slots_to_fill = max(0, target_squad_size - len(selected_details))
        if slots_to_fill > 0:
            candidates = eligible_df[
                (~eligible_df['player_id'].isin([p['player_id'] for p in selected_details]))
            ].sort_values(by='pvs', ascending=False)
            added_count = 0
            for _, player_row in candidates.iterrows():
                if added_count >= slots_to_fill:
                    break
                if player_row['mrb'] <= current_budget:
                    if add_player_to_squad(player_row, False):
                        added_count += 1

        if current_budget > 5 and len(selected_details) == target_squad_size:
            bench_player_details = sorted([p for p in selected_details if not p['is_starter']], key=lambda x: x['pvs'])
            potential_upgrades_pool = eligible_df[~eligible_df['player_id'].isin([p['player_id'] for p in selected_details])].sort_values(by='pvs', ascending=False)
            swapped_in_pass = True
            max_passes = 5
            current_pass = 0
            while swapped_in_pass and current_pass < max_passes and current_budget > 5:
                swapped_in_pass = False
                current_pass += 1
                for i, old_player_detail in enumerate(bench_player_details):
                    if current_budget <= 5:
                        break
                    best_upgrade_for_slot = None
                    best_pvs_gain = 0
                    for _, new_player_row in potential_upgrades_pool[
                        (potential_upgrades_pool['simplified_position'] == old_player_detail['position']) &
                        (potential_upgrades_pool['pvs'] > old_player_detail['pvs'])
                    ].iterrows():
                        cost_to_upgrade = new_player_row['mrb'] - old_player_detail['mrb_cost']
                        if cost_to_upgrade <= current_budget:
                            pvs_gain = new_player_row['pvs'] - old_player_detail['pvs']
                            if pvs_gain > best_pvs_gain:
                                best_pvs_gain = pvs_gain
                                best_upgrade_for_slot = new_player_row
                    if best_upgrade_for_slot is not None:
                        new_player_upg_row = best_upgrade_for_slot
                        cost_to_upgrade = new_player_upg_row['mrb'] - old_player_detail['mrb_cost']
                        old_player_idx_to_remove = -1
                        for idx_sel, sel_p_detail in enumerate(selected_details):
                            if sel_p_detail['player_id'] == old_player_detail['player_id']:
                                old_player_idx_to_remove = idx_sel
                                break
                        if old_player_idx_to_remove != -1:
                            selected_details.pop(old_player_idx_to_remove)
                            add_player_to_squad(new_player_upg_row, False)
                            bench_player_details = sorted([p for p in selected_details if not p['is_starter']], key=lambda x: x['pvs'])
                            potential_upgrades_pool = potential_upgrades_pool[potential_upgrades_pool['player_id'] != new_player_upg_row['player_id']]
                            st.caption(f"Upgraded bench: {df.loc[df['player_id'] == old_player_detail['player_id'], 'Joueur'].iloc[0]} -> {new_player_upg_row['Joueur']} (PVS gain, MRB change: {cost_to_upgrade})")
                            swapped_in_pass = True
                            break
                if not swapped_in_pass:
                    break

        if not selected_details:
            return pd.DataFrame(), {}
        final_squad_ids = [p['player_id'] for p in selected_details]
        final_squad_df = df[df['player_id'].isin(final_squad_ids)].copy()
        details_df = pd.DataFrame(selected_details)
        final_squad_df = pd.merge(final_squad_df, details_df.drop_duplicates(subset=['player_id']), on='player_id', how='left', suffixes=('', '_selection'))
        final_squad_df.rename(columns={'mrb_cost': 'mrb_actual_cost', 'pvs_selection': 'pvs_in_squad'}, inplace=True)
        if 'pvs_in_squad' not in final_squad_df.columns and 'pvs' in final_squad_df.columns:
            final_squad_df['pvs_in_squad'] = final_squad_df['pvs']
        final_squad_df['mrb_actual_cost'] = final_squad_df['mrb_actual_cost'].round().astype(int)
        summary = {
            'total_players': len(final_squad_df),
            'total_cost': final_squad_df['mrb_actual_cost'].sum(),
            'remaining_budget': self.budget - final_squad_df['mrb_actual_cost'].sum(),
            'position_counts': final_squad_df['simplified_position'].value_counts().to_dict(),
            'total_squad_pvs': final_squad_df['pvs_in_squad'].sum(),
            'total_starters_pvs': final_squad_df[final_squad_df['is_starter']]['pvs_in_squad'].sum()
        }
        return final_squad_df, summary

# Main Streamlit App UI
def main():
    st.markdown('<h1 class="main-header">🚀 MPG Auction Strategist v4</h1>', unsafe_allow_html=True)
    strategist = MPGAuctionStrategist()

    # Initialize session state
    if "n_recent" not in st.session_state:
        st.session_state.n_recent = DEFAULT_N_RECENT_GAMES
    if "min_recent_filter" not in st.session_state:
        st.session_state.min_recent_filter = DEFAULT_MIN_RECENT_GAMES_PLAYED
    if 'current_profile_name' not in st.session_state:
        st.session_state.current_profile_name = "Balanced Value"

    # Set profile based on current_profile_name
    if st.session_state.current_profile_name in PREDEFINED_PROFILES:
        profile = PREDEFINED_PROFILES[st.session_state.current_profile_name]
        # Only update settings if profile is not "Custom"
        if st.session_state.current_profile_name != "Custom":
            st.session_state.n_recent = profile.get("n_recent_games", DEFAULT_N_RECENT_GAMES)
            st.session_state.min_recent_filter = profile.get("min_recent_games_played_filter", DEFAULT_MIN_RECENT_GAMES_PLAYED)
            st.session_state.kpi_weights = profile.get("kpi_weights", {})
            st.session_state.mrb_params_per_pos = profile.get("mrb_params_per_pos", {})
        # For "Custom", retain existing session state values (already set by user or defaults)
    else:
        profile = PREDEFINED_PROFILES["Balanced Value"]  # Fallback to a default profile
        st.session_state.current_profile_name = "Balanced Value"
        st.session_state.n_recent = profile.get("n_recent_games", DEFAULT_N_RECENT_GAMES)
        st.session_state.min_recent_filter = profile.get("min_recent_games_played_filter", DEFAULT_MIN_RECENT_GAMES_PLAYED)
        st.session_state.kpi_weights = profile.get("kpi_weights", {})
        st.session_state.mrb_params_per_pos = profile.get("mrb_params_per_pos", {})

    st.session_state.formation_key = DEFAULT_FORMATION
    st.session_state.squad_size = DEFAULT_SQUAD_SIZE

    # Sidebar UI Elements
    st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFr1EWtMR2tHq1FwHnCHqg2uXv1JMLYQlRZw&s", width=100)
    st.sidebar.markdown('<h2 class="section-header" style="margin-top:0;">⚙️ Controls & Settings</h2>', unsafe_allow_html=True)
    uploaded_file = st.sidebar.file_uploader("📁 Upload MPG Ratings File (CSV/Excel)", type=['csv', 'xlsx', 'xls'], help="Joueur, Poste, Club, Cote, %Titu, Indispo?, Gameweeks (D1..D34).")

    if uploaded_file:
        st.sidebar.markdown("---")
        st.sidebar.markdown("#### 🌎 Global Data & Form Parameters")
        n_recent_ui = st.sidebar.number_input("Recent Games Window (N)", min_value=1, max_value=38, value=st.session_state.n_recent, help="For 'Recent Form' KPIs. Avg of games *played* in this window.")
        min_recent_filter_ui = st.sidebar.number_input("Filter: Min Games Played in Recent N Weeks", min_value=0, max_value=n_recent_ui, value=st.session_state.min_recent_filter, help=f"Exclude players with < this in '{n_recent_ui}' recent weeks. 0 = no filter.")
        if n_recent_ui != st.session_state.n_recent or min_recent_filter_ui != st.session_state.min_recent_filter:
            st.session_state.current_profile_name = "Custom"
        st.session_state.n_recent, st.session_state.min_recent_filter = n_recent_ui, min_recent_filter_ui

        st.sidebar.markdown("---")
        st.sidebar.markdown("#### 👥 Squad Building Parameters")
        formation_key_ui = st.sidebar.selectbox("Preferred Starting Formation", options=list(strategist.formations.keys()), index=list(strategist.formations.keys()).index(st.session_state.formation_key))
        target_squad_size_ui = st.sidebar.number_input("Target Total Squad Size", min_value=strategist.squad_minimums_sum_val, max_value=30, value=st.session_state.squad_size)
        if formation_key_ui != st.session_state.formation_key or target_squad_size_ui != st.session_state.squad_size:
            st.session_state.current_profile_name = "Custom"
        st.session_state.formation_key, st.session_state.squad_size = formation_key_ui, target_squad_size_ui

        st.sidebar.markdown("---")
        st.sidebar.markdown("#### 🎨 Settings Profiles")
        profile_names = list(PREDEFINED_PROFILES.keys())

        def apply_profile_settings(profile_name):
            if profile_name != "Custom" and profile_name in PREDEFINED_PROFILES:
                profile = PREDEFINED_PROFILES[profile_name]
                st.session_state.n_recent = profile.get("n_recent_games", st.session_state.n_recent)
                st.session_state.min_recent_filter = profile.get("min_recent_games_played_filter", st.session_state.min_recent_filter)
                st.session_state.kpi_weights = profile.get("kpi_weights", st.session_state.kpi_weights)
                st.session_state.mrb_params_per_pos = profile.get("mrb_params_per_pos", st.session_state.mrb_params_per_pos)
            st.session_state.current_profile_name = profile_name

        selected_profile_name_ui = st.sidebar.selectbox("Select Profile", options=profile_names, index=profile_names.index(st.session_state.current_profile_name), help="Loads predefined settings. Modifying details below sets to 'Custom'.")
        if selected_profile_name_ui != st.session_state.current_profile_name:
            apply_profile_settings(selected_profile_name_ui)
            st.rerun()

        with st.sidebar.expander("📊 KPI Weights (Click to Customize)", expanded=(st.session_state.current_profile_name == "Custom")):
            active_kpi_weights = st.session_state.kpi_weights
            weights_ui = {}
            for pos_key in ['GK', 'DEF', 'MID', 'FWD']:
                st.markdown(f'<h6>{pos_key}</h6>', unsafe_allow_html=True)
                default_pos_w = PREDEFINED_PROFILES["Balanced Value"]["kpi_weights"][pos_key]
                current_pos_w_vals = active_kpi_weights.get(pos_key, default_pos_w)
                weights_ui[pos_key] = {
                    'season_avg': st.slider(f"Season Avg Rating", 0.0, 1.0, float(current_pos_w_vals.get('season_avg', 0)), 0.01, key=f"{pos_key}_wSA_v4"),
                    'season_goals': st.slider(f"Season Goals", 0.0, 1.0, float(current_pos_w_vals.get('season_goals', 0)) if pos_key in ['DEF', 'MID', 'FWD'] else 0.0, 0.01, key=f"{pos_key}_wSG_v4", disabled=pos_key not in ['MID', 'FWD']),
                    'calc_regularity': st.slider(f"Calculated Regularity", 0.0, 1.0, float(current_pos_w_vals.get('calc_regularity', 0)), 0.01, key=f"{pos_key}_wCR_v4", help="Based on starts identified in gameweek data."),
                    'recent_goals': st.slider(f"Recent Goals", 0.0, 1.0, float(current_pos_w_vals.get('recent_goals', 0)) if pos_key in ['MID', 'FWD'] else 0.0, 0.01, key=f"{pos_key}_wRG_v4", disabled=pos_key not in ['MID', 'FWD']),
                    'recent_avg': st.slider(f"Recent Avg Rating", 0.0, 1.0, float(current_pos_w_vals.get('recent_avg', 0)), 0.01, key=f"{pos_key}_wRA_v4"),
                }
            if weights_ui != active_kpi_weights:
                st.session_state.current_profile_name = "Custom"
            st.session_state.kpi_weights = weights_ui

        with st.sidebar.expander("💰 MRB Parameters (Click to Customize)", expanded=(st.session_state.current_profile_name == "Custom")):
            active_mrb_params = st.session_state.mrb_params_per_pos
            mrb_params_ui = {}
            for pos_key in ['GK', 'DEF', 'MID', 'FWD']:
                st.markdown(f'<h6>{pos_key}</h6>', unsafe_allow_html=True)
                default_pos_mrb = PREDEFINED_PROFILES["Balanced Value"]["mrb_params_per_pos"][pos_key]
                current_pos_mrb_vals = active_mrb_params.get(pos_key, default_pos_mrb)
                mrb_params_ui[pos_key] = {
                    'max_proportional_bonus_at_pvs100': st.slider(f"Max Bonus Factor (at PVS 100)", 0.0, 2.0, float(current_pos_mrb_vals.get('max_proportional_bonus_at_pvs100', 0.2)), 0.01, key=f"{pos_key}_mrbMPB_v4", help="Bonus factor if PVS=100 (e.g., 0.5 = 50% bonus). MRB capped at 2x Cote.")
                }
            if mrb_params_ui != active_mrb_params:
                st.session_state.current_profile_name = "Custom"
            st.session_state.mrb_params_per_pos = mrb_params_ui

        if uploaded_file:
            with st.spinner("🧠 Strategizing your optimal squad..."):
                try:
                    df_input_calc = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded_file)
                    df_processed_calc = df_input_calc.copy()
                    df_processed_calc['simplified_position'] = df_processed_calc['Poste'].apply(strategist.simplify_position)
                    df_processed_calc['player_id'] = df_processed_calc.apply(strategist.create_player_id, axis=1)
                    df_processed_calc['Cote'] = pd.to_numeric(df_processed_calc['Cote'], errors='coerce').fillna(1).clip(lower=1).round().astype(int)
                    if 'Indispo ?' not in df_processed_calc.columns:
                        df_processed_calc['Indispo ?'] = False
                    else:
                        df_processed_calc['Indispo ?'] = df_processed_calc['Indispo ?'].astype(str).str.upper().isin(['TRUE', 'OUI', '1', 'YES', 'VRAI'])
                    df_kpis = strategist.calculate_kpis(df_processed_calc, st.session_state.n_recent)
                    df_norm_kpis = strategist.normalize_kpis(df_kpis)
                    df_pvs = strategist.calculate_pvs(df_norm_kpis, st.session_state.kpi_weights)
                    df_mrb = strategist.calculate_mrb(df_pvs, st.session_state.mrb_params_per_pos)
                    squad_df_result, squad_summary_result = strategist.select_squad(
                        df_mrb, st.session_state.formation_key, st.session_state.squad_size, st.session_state.min_recent_filter
                    )
                    st.session_state['df_for_display_final'] = df_mrb
                    st.session_state['squad_df_result_final'] = squad_df_result
                    st.session_state['squad_summary_result_final'] = squad_summary_result
                    st.session_state['selected_formation_key_display_final'] = st.session_state.formation_key
                except Exception as e:
                    st.error(f"💥 Error during dynamic calculation: {str(e)}")

            if 'squad_df_result_final' in st.session_state and st.session_state['squad_df_result_final'] is not None and \
               not st.session_state['squad_df_result_final'].empty:
                col_main_results, col_summary = st.columns([3, 1])
                with col_main_results:
                    st.markdown('<h2 class="section-header">🏆 Suggested Squad</h2>', unsafe_allow_html=True)
                    sdf = st.session_state['squad_df_result_final'].copy()
                    int_cols = ['mrb_actual_cost', 'Cote', 'recent_goals', 'season_goals']
                    for col in int_cols:
                        if col in sdf.columns:
                            sdf[col] = pd.to_numeric(sdf[col], errors='coerce').fillna(0).round().astype(int)
                    squad_cols = ['Joueur', 'Club', 'simplified_position', 'pvs_in_squad', 'Cote', 'mrb_actual_cost', 'season_avg_rating', 'season_goals', 'calc_regularity_pct', 'recent_goals', 'recent_avg_rating', 'value_per_cost', 'is_starter']
                    squad_cols_exist = [col for col in squad_cols if col in sdf.columns]
                    sdf = sdf[squad_cols_exist]
                    sdf.rename(columns={
                        'Joueur': 'Player',
                        'simplified_position': 'Pos',
                        'pvs_in_squad': 'PVS',
                        'Cote': 'Cote',
                        'mrb_actual_cost': 'Suggested Bid',
                        'season_avg_rating': 'Average',
                        'season_goals': 'Goals',
                        'calc_regularity_pct': '% played',
                        'recent_goals': 'Rec.G',
                        'recent_avg_rating': 'Rec.AvgR',
                        'value_per_cost': 'Val/MRB',
                        'is_starter': 'Starter'
                    }, inplace=True)
                    float_cols_squad = ['PVS', 'Rec.AvgR', 'Sea.AvgR', 'Reg.% (File)', 'Reg.% (Calc)', 'Val/MRB']
                    for col in float_cols_squad:
                        if col in sdf.columns:
                            sdf[col] = pd.to_numeric(sdf[col], errors='coerce').fillna(0.0).round(2)
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
                        for pos_cat in pos_order:
                            count = summary.get('position_counts', {}).get(pos_cat, 0)
                            min_req = strategist.squad_minimums.get(pos_cat, 0)
                            st.markdown(f"• **{pos_cat}:** {count} (Min: {min_req})")
                    else:
                        st.warning("Squad summary unavailable.")

                st.markdown('<hr><h2 class="section-header">📋 Full Player Database & Values</h2>', unsafe_allow_html=True)
                if 'df_for_display_final' in st.session_state and st.session_state['df_for_display_final'] is not None:
                    df_full = st.session_state['df_for_display_final'].copy()
                    int_cols_full = ['Cote', 'mrb', 'recent_goals', 'season_goals', 'recent_games_played_count', 'games_started_season', 'total_season_gws_considered']
                    for col in int_cols_full:
                        if col in df_full.columns:
                            df_full[col] = pd.to_numeric(df_full[col], errors='coerce').fillna(0).round().astype(int)
                    all_stats_cols = ['Joueur', 'Club', 'simplified_position', 'pvs', 'Cote', 'mrb', 'Indispo ?', 'season_avg_rating', 'season_goals',
                                      'calc_regularity_pct', 'recent_goals', 'recent_avg_rating', 'value_per_cost',
                                       'games_started_season', 'recent_games_played_count']
                    df_full = df_full[[col for col in all_stats_cols if col in df_full.columns]]
                    df_full.rename(columns={
                        'Joueur': 'Player',
                        'simplified_position': 'Pos',
                        'pvs': 'PVS',
                        'Cote': 'Cote',
                        'mrb': 'Suggested Bid',
                        'Indispo ?': 'Unavail.',
                        'season_avg_rating': 'Average',
                        'season_goals': 'Goals',
                        'calc_regularity_pct': '% Played',
                        'recent_goals': 'Rec.G',
                        'recent_avg_rating': 'Rec.AvgR',
                        'value_per_cost': 'Val/MRB',
                        'games_started_season': 'Sea.Start',
                        'recent_games_played_count': 'Rec.Plyd'
                    }, inplace=True)
                    float_cols_full = ['PVS', 'Val/MRB', 'Rec.AvgR', 'Sea.AvgR', 'Reg.%File', 'Reg.%Calc',
                                       'N.RecAvg', 'N.SeaAvg', 'N.RegFile', 'N.RegCalc', 'N.RecG', 'N.SeaG']
                    for col in float_cols_full:
                        if col in df_full.columns:
                            df_full[col] = pd.to_numeric(df_full[col], errors='coerce').fillna(0.0).round(2)
                    search_all = st.text_input("🔍 Search All Players:", key="search_all_v4")
                    if search_all:
                        df_full = df_full[df_full.apply(lambda r: r.astype(str).str.contains(search_all, case=False, na=False).any(), axis=1)]
                    st.dataframe(df_full.sort_values(by='PVS', ascending=False), use_container_width=True, hide_index=True, height=600)
                    st.download_button(
                        label="📥 Download Full Analysis (CSV)",
                        data=df_full.to_csv(index=False).encode('utf-8'),
                        file_name="mpg_full_player_analysis_v4.csv",
                        mime="text/csv",
                        key="download_v4"
                    )
            elif not uploaded_file:
                pass
            elif 'squad_df_result_final' not in st.session_state and uploaded_file:
                st.info("📊 Adjust settings in the sidebar. Results update dynamically when inputs change.")
        else:
            st.info("👈 Upload your MPG ratings file to begin.")

if __name__ == "__main__":
    main()
