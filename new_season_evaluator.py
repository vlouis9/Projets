import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Set
from io import BytesIO

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="MPG Hybrid Season Strategist",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {font-size: 2.5rem; font-weight: bold; color: #004080; text-align: center; margin-bottom: 2rem; font-family: 'Roboto', sans-serif;}
    .section-header {font-size: 1.4rem; font-weight: bold; color: #006847; margin-top: 1.5rem; margin-bottom: 1rem; border-bottom: 2px solid #006847; padding-bottom: 0.3rem;}
    .stButton>button {background-color: #004080; color: white; font-weight: bold; border-radius: 0.3rem; padding: 0.4rem 0.8rem; border: none; width: 100%;}
    .stButton>button:hover {background-color: #003060; color: white;}
    .stExpander {border: 1px solid #e0e0e0; border-radius: 0.3rem; margin-bottom: 0.5rem;}
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
    },
    "Focus on High Potential": {
        "team_rank_weight": 0.15,
        "kpi_weights": {
            'GK':  {KPI_PERFORMANCE: 0.20, KPI_POTENTIAL: 0.50, KPI_REGULARITY: 0.30, KPI_GOALS: 0.00},
            'DEF': {KPI_PERFORMANCE: 0.20, KPI_POTENTIAL: 0.50, KPI_REGULARITY: 0.20, KPI_GOALS: 0.10},
            'MID': {KPI_PERFORMANCE: 0.15, KPI_POTENTIAL: 0.50, KPI_REGULARITY: 0.15, KPI_GOALS: 0.20},
            'FWD': {KPI_PERFORMANCE: 0.15, KPI_POTENTIAL: 0.45, KPI_REGULARITY: 0.10, KPI_GOALS: 0.30}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.5}, 'DEF': {'max_proportional_bonus_at_pvs100': 0.6},
            'MID': {'max_proportional_bonus_at_pvs100': 0.8}, 'FWD': {'max_proportional_bonus_at_pvs100': 1.0}
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
        rdf['pvs'] = 0.0

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
                player_pvs.loc[mask] = (pos_pvs_sum / total_weight * 100) if total_weight != 1.0 else pos_pvs_sum
        
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
                cote = int(row['Cote']); pvs_player = float(row['pvs'])
                pvs_scaled = pvs_player / 100.0
                bonus_factor = pvs_scaled * max_prop_bonus
                mrb_float = cote * (1 + bonus_factor)
                mrb_capped = min(mrb_float, float(cote * 2))
                final_mrb = max(float(cote), mrb_capped)
                return int(round(final_mrb))
            rdf.loc[mask, 'mrb'] = rdf.loc[mask].apply(_calc_mrb_player, axis=1)
        rdf['mrb'] = rdf['mrb'].astype(int)
        safe_mrb = rdf['mrb'].replace(0, 1).astype(float)
        rdf['value_per_cost'] = rdf['pvs'] / safe_mrb
        rdf['value_per_cost'].fillna(0, inplace=True)
        return rdf

    def select_squad(self, df_evaluated_players: pd.DataFrame, formation_key: str, target_squad_size: int) -> Tuple[pd.DataFrame, Dict]:
        # This is the full, robust squad selection logic from your previous historical data app
        eligible_df = df_evaluated_players.drop_duplicates(subset=['player_id']).copy()
        eligible_df['mrb'] = eligible_df['mrb'].astype(int)
        squad = []

        def get_squad_ids(): return {p['player_id'] for p in squad}
        def get_pos_counts():
            counts = {pos_key: 0 for pos_key in self.squad_minimums.keys()}
            for p in squad: counts[p['pos']] = counts.get(p['pos'], 0) + 1
            return counts

        def add_player(player_row, is_starter):
            if player_row['player_id'] in get_squad_ids(): return False
            if player_row['simplified_position'] == 'GK' and get_pos_counts().get('GK', 0) >= 2: return False
            squad.append({'player_id': player_row['player_id'], 'mrb': int(player_row['mrb']),'pvs': float(player_row['pvs']), 'pos': player_row['simplified_position'],'is_starter': is_starter})
            return True

        def remove_player(player_id):
            nonlocal squad
            initial_len = len(squad)
            squad = [p for p in squad if p['player_id'] != player_id]
            return len(squad) < initial_len

        all_players_sorted_pvs = eligible_df.sort_values(by='pvs', ascending=False)
        starters_map = self.formations[formation_key].copy()
        for _, row in all_players_sorted_pvs.iterrows():
            if row['player_id'] not in get_squad_ids() and starters_map.get(row['simplified_position'], 0) > 0:
                if add_player(row, True): starters_map[row['simplified_position']] -= 1
        
        for pos, min_needed in self.squad_minimums.items():
            while get_pos_counts().get(pos, 0) < min_needed:
                candidate = all_players_sorted_pvs[(all_players_sorted_pvs['simplified_position'] == pos) & (~all_players_sorted_pvs['player_id'].isin(get_squad_ids()))].head(1)
                if candidate.empty or not add_player(candidate.iloc[0], False): break
        
        while len(squad) < target_squad_size:
            candidate = all_players_sorted_pvs[~all_players_sorted_pvs['player_id'].isin(get_squad_ids())].head(1)
            if candidate.empty or not add_player(candidate.iloc[0], False): break

        current_mrb = sum(p['mrb'] for p in squad)
        for _ in range(target_squad_size * 2):
            if current_mrb <= self.budget: break
            best_downgrade = None
            for p_old in sorted(squad, key=lambda x: x['mrb'], reverse=True):
                replacements = eligible_df[(eligible_df['simplified_position'] == p_old['pos']) & (~eligible_df['player_id'].isin(get_squad_ids() | {p_old['player_id']})) & (eligible_df['mrb'] < p_old['mrb'])]
                for _, p_new in replacements.iterrows():
                    score = (p_old['mrb'] - p_new['mrb']) - (p_old['pvs'] - p_new['pvs'])
                    if best_downgrade is None or score > best_downgrade[2]:
                        best_downgrade = (p_old, p_new, score)
            if best_downgrade:
                old, new, _ = best_downgrade
                remove_player(old['player_id'])
                add_player(new, old['is_starter'])
                current_mrb = sum(p['mrb'] for p in squad)
            else: break
        
        for _ in range(target_squad_size):
            budget_left = self.budget - current_mrb
            if budget_left <= 5: break
            best_upgrade = None
            for p_old in sorted(squad, key=lambda x: x['pvs']):
                replacements = eligible_df[(eligible_df['simplified_position'] == p_old['pos']) & (~eligible_df['player_id'].isin(get_squad_ids() | {p_old['player_id']})) & (eligible_df['pvs'] > p_old['pvs']) & (eligible_df['mrb'] <= p_old['mrb'] + budget_left)]
                for _, p_new in replacements.iterrows():
                    score = (p_new['pvs'] - p_old['pvs']) / ((p_new['mrb'] - p_old['mrb']) + 1)
                    if best_upgrade is None or score > best_upgrade[2]:
                        best_upgrade = (p_old, p_new, score)
            if best_upgrade:
                old, new, _ = best_upgrade
                remove_player(old['player_id'])
                add_player(new, old['is_starter'])
                current_mrb = sum(p['mrb'] for p in squad)
            else: break

        if not squad: return pd.DataFrame(), {}
        final_squad_df = eligible_df[eligible_df['player_id'].isin(get_squad_ids())].copy()
        details_df = pd.DataFrame(squad).rename(columns={'mrb': 'mrb_actual_cost', 'pvs':'pvs_in_squad'})
        final_squad_df = pd.merge(final_squad_df, details_df[['player_id', 'mrb_actual_cost', 'pvs_in_squad', 'is_starter']], on='player_id', how='left')
        
        final_total_mrb = final_squad_df['mrb_actual_cost'].sum()
        summary = {'total_players': len(final_squad_df), 'total_cost': int(final_total_mrb),'remaining_budget': int(self.budget - final_total_mrb), 'position_counts': final_squad_df['simplified_position'].value_counts().to_dict(),'total_squad_pvs': round(final_squad_df['pvs_in_squad'].sum(), 2),'total_starters_pvs': round(final_squad_df[final_squad_df['is_starter']]['pvs_in_squad'].sum(), 2)}
        return final_squad_df, summary

# --- Data Processing & Logic Functions ---
@st.cache_data
def load_and_reconcile_players(hist_file, new_season_file):
    if hist_file is None or new_season_file is None: return None, None, None, None
    try:
        df_hist = pd.read_excel(hist_file) if hist_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(hist_file)
        df_new = pd.read_excel(new_season_file) if new_season_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(new_season_file)
        
        for df in [df_hist, df_new]:
            df['simplified_position'] = df['Poste'].apply(MPGAuctionStrategist.simplify_position)
            df['player_id'] = df.apply(MPGAuctionStrategist.create_player_id, axis=1)

        hist_ids, new_ids = set(df_hist['player_id']), set(df_new['player_id'])
        return df_hist, df_new, hist_ids.intersection(new_ids), new_ids - hist_ids
    except Exception as e:
        st.error(f"Error loading files: {e}"); return None, None, None, None

def extract_rating(rating_str):
    if pd.isna(rating_str) or str(rating_str).strip() in ['', '0']: return None, 0
    clean_str = re.sub(r'[()\*]', '', str(rating_str).strip())
    try: return float(clean_str), str(rating_str).count('*')
    except (ValueError, TypeError): return None, 0

@st.cache_data
def calculate_historical_kpis(df_hist, returning_ids):
    df = df_hist[df_hist['player_id'].isin(returning_ids)].copy()
    gw_cols = [col for col in df.columns if col.startswith('D')]
    
    kpi_data = []
    for _, row in df.iterrows():
        ratings, goals = [], 0
        for gw in gw_cols:
            r, g = extract_rating(row.get(gw))
            if r is not None: ratings.append(r); goals += g
        
        kpi_data.append({
            'player_id': row['player_id'],
            KPI_PERFORMANCE: np.mean(ratings) if ratings else 0,
            KPI_POTENTIAL: np.mean(sorted(ratings, reverse=True)[:5]) if ratings else 0,
            KPI_REGULARITY: (len(ratings) / len(gw_cols) * 100) if gw_cols else 0,
            KPI_GOALS: goals
        })
    
    kpi_df = pd.DataFrame(kpi_data)
    for kpi in PLAYER_KPI_COLUMNS:
        max_val = kpi_df[kpi].max()
        kpi_df[f"norm_{kpi}"] = (kpi_df[kpi] / max_val * 100) if max_val > 0 else 0
    return kpi_df

# --- Main App ---
def main():
    st.markdown('<h1 class="main-header">🌟 MPG Hybrid Season Strategist</h1>', unsafe_allow_html=True)
    strategist = MPGAuctionStrategist()

    st.sidebar.markdown('<h2 class="section-header" style="margin-top:0;">📁 File Inputs</h2>', unsafe_allow_html=True)
    hist_file = st.sidebar.file_uploader("1. Upload Historical Season Data", type=['csv', 'xlsx', 'xls'])
    new_season_file = st.sidebar.file_uploader("2. Upload New Season Player List", type=['csv', 'xlsx', 'xls'])

    if not hist_file or not new_season_file:
        st.info("👈 Please upload both historical data and the new season player list to begin."); return

    df_hist, df_new, returning_ids, new_player_ids = load_and_reconcile_players(hist_file, new_season_file)
    if df_hist is None: return
    
    df_returning_kpis = calculate_historical_kpis(df_hist, returning_ids)
    
    if 'team_tiers' not in st.session_state: st.session_state.team_tiers = {'Winner': [], 'European': [], 'Average': sorted(list(df_new['Club'].unique())), 'Relegation': []}
    if 'new_player_kpis' not in st.session_state: st.session_state.new_player_kpis = {}
    if 'current_profile_name' not in st.session_state: st.session_state.current_profile_name = "Balanced Value"

    st.markdown('<h2 class="section-header">1. Team Ranking Setup</h2>', unsafe_allow_html=True)
    all_clubs = sorted(list(df_new['Club'].unique())); assigned_clubs = set()
    tier_cols = st.columns(4); tier_names = ["Winner", "European", "Average", "Relegation"]
    for i, tier in enumerate(tier_names):
        with tier_cols[i]:
            available_clubs = [c for c in all_clubs if c not in assigned_clubs]
            current_selection = [c for c in st.session_state.team_tiers.get(tier, []) if c in all_clubs]
            selections = st.multiselect(f"**{tier} Tier**", options=available_clubs + list(set(current_selection)), default=current_selection, key=f"tier_{tier}")
            st.session_state.team_tiers[tier] = selections
            for s in selections: assigned_clubs.add(s)

    st.markdown('<hr><h2 class="section-header">2. New Player KPI Setup</h2>', unsafe_allow_html=True)
    df_new_players = df_new[df_new['player_id'].isin(new_player_ids)].copy()
    if df_new_players.empty: st.info("No new players identified.")
    else:
        kpi_ranges = {pos: {kpi: (df_returning_kpis[df_returning_kpis['player_id'].str.contains(f"_{pos}_")][f"norm_{kpi}"].min(), df_returning_kpis[df_returning_kpis['player_id'].str.contains(f"_{pos}_")][f"norm_{kpi}"].max()) if not df_returning_kpis[df_returning_kpis['player_id'].str.contains(f"_{pos}_")].empty else (0, 100) for kpi in PLAYER_KPI_COLUMNS} for pos in ['GK', 'DEF', 'MID', 'FWD']}
        
        for _, player in df_new_players.iterrows():
            player_id = player['player_id']; pos = player['simplified_position']
            st.markdown(f"--- \n**{player['Joueur']}** ({player['Club']} - {pos} - Cote: {player['Cote']})")
            if player_id not in st.session_state.new_player_kpis: st.session_state.new_player_kpis[player_id] = {}
            kpi_cols = st.columns(4)
            for i, kpi in enumerate(PLAYER_KPI_COLUMNS):
                with kpi_cols[i]:
                    min_val, max_val = kpi_ranges.get(pos, {}).get(kpi, (0, 100))
                    options = [round(min_val + ((max_val - min_val) * p), 2) for p in [0, 0.25, 0.5, 0.75, 1.0]]
                    default_value = st.session_state.new_player_kpis[player_id].get(f"norm_{kpi}", options[2])
                    val = st.select_slider(f"{kpi.replace('Estimation','')}", options=options, value=default_value, key=f"{player_id}_{kpi}")
                    st.session_state.new_player_kpis[player_id][f"norm_{kpi}"] = val
        
        if st.button("Create Download File for New Player KPIs"):
            new_kpi_list = []
            for player_id, kpis in st.session_state.new_player_kpis.items():
                player_info = df_new[df_new['player_id'] == player_id].iloc[0]
                entry = {'Joueur': player_info['Joueur'], 'Club': player_info['Club'], 'Poste': player_info['Poste']}
                entry.update({k.replace('norm_', ''): v for k, v in kpis.items()})
                new_kpi_list.append(entry)
            if new_kpi_list:
                df_to_save = pd.DataFrame(new_kpi_list)
                output = BytesIO()
                with pd.ExcelWriter(output, engine='xlsxwriter') as writer: df_to_save.to_excel(writer, index=False, sheet_name='NewPlayerKPIs')
                st.session_state.new_kpi_excel_data = output.getvalue()
    
    if 'new_kpi_excel_data' in st.session_state:
        st.download_button(label="📥 Download New Player KPIs (Excel)", data=st.session_state.new_kpi_excel_data, file_name="new_player_kpis.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    # --- Sidebar Controls for Final Calculation ---
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("### 3. PVS & Squad Parameters")
    
    profile_name = st.sidebar.selectbox("Select Profile", options=list(PREDEFINED_PROFILES.keys()), key="profile_selector")
    
    if profile_name != "Custom":
        st.session_state.team_rank_weight = PREDEFINED_PROFILES[profile_name]["team_rank_weight"]
        st.session_state.kpi_weights = PREDEFINED_PROFILES[profile_name]["kpi_weights"]
        st.session_state.mrb_params_per_pos = PREDEFINED_PROFILES[profile_name]["mrb_params_per_pos"]
    
    st.session_state.team_rank_weight = st.sidebar.slider("Team Ranking KPI Weight", 0.0, 1.0, st.session_state.get('team_rank_weight', 0.2), 0.05, help="Weight of club tier in total PVS.")
    
    with st.sidebar.expander("Customize Player KPI Weights"):
        # UI for kpi_weights
        pass # Placeholder for detailed sliders
    with st.sidebar.expander("Customize MRB Parameters"):
        # UI for mrb_params_per_pos
        pass # Placeholder for detailed sliders

    st.session_state.formation_key = st.sidebar.selectbox("Preferred Formation", options=list(strategist.formations.keys()), index=0)
    st.session_state.squad_size = st.sidebar.number_input("Target Squad Size", min_value=18, max_value=30, value=DEFAULT_SQUAD_SIZE)

    if st.sidebar.button("🚀 Generate Optimal Squad", type="primary"):
        # Consolidate KPIs
        new_kpis_df = pd.DataFrame([{'player_id': pid, **kpis} for pid, kpis in st.session_state.new_player_kpis.items()])
        all_kpis_df = pd.concat([df_returning_kpis, new_kpis_df], ignore_index=True)
        # Add Team Ranking
        tier_map = {100: "Winner", 75: "European", 50: "Average", 25: "Relegation"}
        club_to_score = {club: score for score, tier in tier_map.items() for club in st.session_state.team_tiers[tier]}
        df_new['Cote'] = pd.to_numeric(df_new['Cote'], errors='coerce').fillna(1).clip(lower=1)
        df_merged = pd.merge(df_new[df_new['player_id'].isin(returning_ids.union(new_player_ids))], all_kpis_df, on='player_id', how='left')
        df_merged[KPI_TEAM_RANK] = df_merged['Club'].map(club_to_score).fillna(50)
        
        # Calculate and select
        df_pvs = strategist.calculate_pvs(df_merged, st.session_state.team_rank_weight, st.session_state.get('kpi_weights', {}))
        df_mrb = strategist.calculate_mrb(df_pvs, st.session_state.get('mrb_params_per_pos', {}))
        squad_df, summary = strategist.select_squad(df_mrb, st.session_state.formation_key, st.session_state.squad_size)
        st.session_state.squad_df_result, st.session_state.squad_summary_result = squad_df, summary

    if 'squad_df_result' in st.session_state:
        st.markdown('<hr><h2 class="section-header">🏆 Suggested Squad</h2>', unsafe_allow_html=True)
        st.dataframe(st.session_state.squad_df_result)
        st.markdown('<h2 class="section-header">📈 Squad Summary</h2>', unsafe_allow_html=True)
        st.json(st.session_state.squad_summary_result)

if __name__ == "__main__":
    main()
