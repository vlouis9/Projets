import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Set

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="MPG Hybrid Strategist v9.0",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #004080; text-align: center; margin-bottom: 2rem; font-family: 'Roboto', sans-serif;}
    .section-header {font-size: 1.4rem; font-weight: bold; color: #006847; margin-top: 1.5rem; margin-bottom: 1rem; border-bottom: 2px solid #006847; padding-bottom: 0.3rem;}
    .stButton>button {background-color: #004080; color: white; font-weight: bold; border-radius: 0.3rem; padding: 0.4rem 0.8rem; border: none; width: 100%;}
    .stButton>button:hover {background-color: #003060; color: white;}
</style>
""", unsafe_allow_html=True)


# --- KPI & Profile Definitions ---
KPI_PERFORMANCE = "PerformanceEstimation"
KPI_POTENTIAL = "PotentialEstimation"
KPI_REGULARITY = "RegularityEstimation"
KPI_GOALS = "GoalsEstimation"
KPI_TEAM_RANK = "TeamRanking"
PLAYER_KPI_COLUMNS = [KPI_PERFORMANCE, KPI_POTENTIAL, KPI_REGULARITY, KPI_GOALS]

DEFAULT_FORMATION = "4-4-2"
DEFAULT_SQUAD_SIZE = 20

PREDEFINED_PROFILES = {
    "Custom": "custom",
    "Balanced Value": {
        "team_rank_weight": 0.20,
        "kpi_weights": {
            'GK':  {KPI_PERFORMANCE: 0.50, KPI_POTENTIAL: 0.10, KPI_REGULARITY: 0.40, KPI_GOALS: 0.00},
            'DEF': {KPI_PERFORMANCE: 0.40, KPI_POTENTIAL: 0.15, KPI_REGULARITY: 0.35, KPI_GOALS: 0.10},
            'MID': {KPI_PERFORMANCE: 0.35, KPI_POTENTIAL: 0.20, KPI_REGULARITY: 0.25, KPI_GOALS: 0.20},
            'FWD': {KPI_PERFORMANCE: 0.30, KPI_POTENTIAL: 0.20, KPI_REGULARITY: 0.20, KPI_GOALS: 0.30}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.3}, 'DEF': {'max_proportional_bonus_at_pvs100': 0.4},
            'MID': {'max_proportional_bonus_at_pvs100': 0.6}, 'FWD': {'max_proportional_bonus_at_pvs100': 0.8}
        }
    }
}

class MPGAuctionStrategist:
    def __init__(self):
        self.formations = {"4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}, "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},"3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2}, "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},"4-5-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1}, "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},"5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1}}
        self.squad_minimums = {"GK": 2, "DEF": 6, "MID": 6, "FWD": 4}
        self.budget = 500

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
    def calculate_pvs(df: pd.DataFrame, team_rank_weight: float, weights: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        rdf = df.copy()
        player_pvs = pd.Series(0.0, index=rdf.index)
        for pos, pos_weights in weights.items():
            mask = rdf['simplified_position'] == pos
            if not mask.any(): continue
            pos_pvs_sum = pd.Series(0.0, index=rdf.loc[mask].index)
            total_weight = sum(pos_weights.values())
            for kpi_col_base in PLAYER_KPI_COLUMNS:
                norm_kpi_col = f"norm_{kpi_col_base}"
                weight = pos_weights.get(kpi_col_base, 0)
                if norm_kpi_col in rdf.columns:
                    pos_pvs_sum += rdf.loc[mask, norm_kpi_col].fillna(0) * weight
            if total_weight > 0:
                player_pvs.loc[mask] = (pos_pvs_sum / total_weight * 100) if abs(total_weight - 1.0) > 1e-6 else pos_pvs_sum
        player_pvs = player_pvs.clip(0, 100)
        team_rank_score = rdf[KPI_TEAM_RANK].clip(0, 100)
        rdf['pvs'] = ((player_pvs * (1 - team_rank_weight)) + (team_rank_score * team_rank_weight)).clip(0, 100)
        return rdf

    @staticmethod
    def calculate_mrb(df: pd.DataFrame, mrb_params_per_pos: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        rdf = df.copy()
        rdf['mrb'] = rdf['Cote']
        for pos_simplified, params in mrb_params_per_pos.items():
            mask = rdf['simplified_position'] == pos_simplified
            if not mask.any(): continue
            max_prop_bonus = params.get('max_proportional_bonus_at_pvs100', 0.5)
            def _calc_mrb_player(row):
                cote, pvs_player = int(row['Cote']), float(row['pvs'])
                bonus_factor = (pvs_player / 100.0) * max_prop_bonus
                return int(round(max(float(cote), min(cote * (1 + bonus_factor), float(cote * 2)))))
            rdf.loc[mask, 'mrb'] = rdf.loc[mask].apply(_calc_mrb_player, axis=1)
        rdf['mrb'] = rdf['mrb'].astype(int)
        rdf['value_per_cost'] = rdf['pvs'] / rdf['mrb'].replace(0, 1).astype(float)
        return rdf

    def select_squad(self, df_evaluated_players: pd.DataFrame, formation_key: str, target_squad_size: int) -> Tuple[pd.DataFrame, Dict]:
        eligible_df = df_evaluated_players.drop_duplicates(subset=['player_id']).copy()
        eligible_df['mrb'] = eligible_df['mrb'].astype(int)
        squad = []

        def get_squad_ids(): return {p['player_id'] for p in squad}
        def get_pos_counts():
            counts = {pos: 0 for pos in self.squad_minimums}
            for p in squad: counts[p['pos']] = counts.get(p['pos'], 0) + 1
            return counts
        def add_player(p_row, is_starter):
            if p_row['player_id'] in get_squad_ids(): return False
            if p_row['simplified_position'] == 'GK' and get_pos_counts().get('GK', 0) >= 2: return False
            squad.append({'player_id': p_row['player_id'], 'mrb': int(p_row['mrb']), 'pvs': float(p_row['pvs']), 'pos': p_row['simplified_position'], 'is_starter': is_starter})
            return True
        def remove_player(p_id):
            nonlocal squad
            squad = [p for p in squad if p['player_id'] != p_id]

        all_players_sorted_pvs = eligible_df.sort_values(by='pvs', ascending=False)
        starters_map = self.formations[formation_key].copy()
        for _, row in all_players_sorted_pvs.iterrows():
            if starters_map.get(row['simplified_position'], 0) > 0 and add_player(row, True):
                starters_map[row['simplified_position']] -= 1

        for pos, min_needed in self.squad_minimums.items():
            while get_pos_counts().get(pos, 0) < min_needed:
                candidate = all_players_sorted_pvs[(all_players_sorted_pvs['simplified_position'] == pos) & (~all_players_sorted_pvs['player_id'].isin(get_squad_ids()))].head(1)
                if candidate.empty or not add_player(candidate.iloc[0], False): break
        
        # --- REBUILT FILL-UP LOGIC ---
        while len(squad) < target_squad_size:
            available_pool = all_players_sorted_pvs[~all_players_sorted_pvs['player_id'].isin(get_squad_ids())]
            if available_pool.empty:
                break # No more players left in the league to add
            
            player_added = False
            for _, candidate in available_pool.iterrows():
                if add_player(candidate, False):
                    player_added = True
                    break # Player added, restart the while loop
            
            if not player_added:
                break # Went through all available players and couldn't add any

        current_mrb = sum(p['mrb'] for p in squad)
        for _ in range(target_squad_size * 2): # Budget optimization loop
            if current_mrb <= self.budget: break
            best_downgrade = None
            for p_old in sorted(squad, key=lambda x: x['mrb'], reverse=True):
                if p_old['is_starter']: continue # Don't downgrade starters if possible
                replacements = eligible_df[(eligible_df['simplified_position'] == p_old['pos']) & (~eligible_df['player_id'].isin(get_squad_ids() - {p_old['player_id']})) & (eligible_df['mrb'] < p_old['mrb'])].sort_values('pvs', ascending=False)
                if replacements.empty: continue
                p_new = replacements.iloc[0]
                score = (p_old['mrb'] - p_new['mrb']) - (p_old['pvs'] - p_new['pvs']) * 0.5
                if best_downgrade is None or score > best_downgrade[2]:
                    best_downgrade = (p_old, p_new.to_dict(), score)
            if best_downgrade:
                old, new_dict, _ = best_downgrade
                remove_player(old['player_id'])
                add_player(pd.Series(new_dict), old['is_starter'])
                current_mrb = sum(p['mrb'] for p in squad)
            else: break
        
        if not squad: return pd.DataFrame(), {}
        final_df = eligible_df[eligible_df['player_id'].isin(get_squad_ids())].copy()
        details_df = pd.DataFrame(squad).rename(columns={'mrb': 'mrb_actual_cost', 'pvs':'pvs_in_squad'})
        final_df = pd.merge(final_df, details_df, on='player_id')
        summary = {'total_players': len(final_df), 'total_cost': int(final_df['mrb_actual_cost'].sum()),'remaining_budget': int(self.budget - final_df['mrb_actual_cost'].sum()), 'position_counts': final_df['simplified_position'].value_counts().to_dict(),'total_squad_pvs': round(final_df['pvs_in_squad'].sum(), 2),'total_starters_pvs': round(final_df[final_df['is_starter']]['pvs_in_squad'].sum(), 2)}
        return final_df, summary

# --- Data Processing Functions ---
@st.cache_data
def load_and_reconcile_players(hist_file, new_season_file):
    try:
        df_hist = pd.read_excel(hist_file) if hist_file.name.endswith('.xlsx') else pd.read_csv(hist_file)
        df_new = pd.read_excel(new_season_file) if new_season_file.name.endswith('.xlsx') else pd.read_csv(new_season_file)
        for df in [df_hist, df_new]:
            df['simplified_position'] = df['Poste'].apply(MPGAuctionStrategist.simplify_position)
            df['player_id'] = df.apply(MPGAuctionStrategist.create_player_id, axis=1)
        hist_ids, new_ids = set(df_hist['player_id']), set(df_new['player_id'])
        return df_hist, df_new, hist_ids.intersection(new_ids), new_ids - hist_ids
    except Exception as e: st.error(f"Error loading files: {e}"); return None, None, None, None

def extract_rating(rating_str):
    if pd.isna(rating_str) or str(rating_str).strip() in ['', '0']: return None
    clean_str = re.sub(r'[()\*]', '', str(rating_str).strip())
    try: return float(clean_str)
    except (ValueError, TypeError): return None

@st.cache_data
def calculate_historical_kpis(df_hist, returning_ids):
    df = df_hist[df_hist['player_id'].isin(returning_ids)].copy()
    gw_cols = [col for col in df.columns if str(col).startswith('D')]
    kpi_data = []
    for _, row in df.iterrows():
        ratings_with_goals = [(extract_rating(row.get(gw)), str(row.get(gw, '')).count('*')) for gw in gw_cols]
        valid_ratings = [r[0] for r in ratings_with_goals if r[0] is not None]
        if valid_.
