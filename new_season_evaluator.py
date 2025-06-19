import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Set

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="MPG Hybrid Strategist v13.0",
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
POSITIONS = ['GK', 'DEF', 'MID', 'FWD']

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
        rdf['pvs_kpi'] = 0.0
        for pos in POSITIONS:
            mask = rdf['simplified_position'] == pos
            if not mask.any(): continue
            pos_weights = weights.get(pos, {})
            total_weight = sum(pos_weights.values())
            if total_weight > 0:
                weighted_sum = sum(rdf.loc[mask, f"norm_{kpi}"] * w for kpi, w in pos_weights.items() if f"norm_{kpi}" in rdf.columns)
                # FIX: Removed erroneous * 100, PVS is now a proper weighted average
                rdf.loc[mask, 'pvs_kpi'] = weighted_sum / total_weight
        
        rdf['pvs'] = ((rdf['pvs_kpi'] * (1 - team_rank_weight)) + (rdf[KPI_TEAM_RANK] * team_rank_weight)).clip(0, 100)
        return rdf

    @staticmethod
    def calculate_mrb(df: pd.DataFrame, mrb_params_per_pos: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        rdf = df.copy()
        rdf['mrb'] = rdf['Cote']
        for pos, params in mrb_params_per_pos.items():
            mask = rdf['simplified_position'] == pos
            if not mask.any(): continue
            max_prop_bonus = params.get('max_proportional_bonus_at_pvs100', 0.5)
            bonus_factor = (rdf.loc[mask, 'pvs'] / 100.0) * max_prop_bonus
            rdf.loc[mask, 'mrb'] = np.round(np.maximum(rdf.loc[mask, 'Cote'], np.minimum(rdf.loc[mask, 'Cote'] * (1 + bonus_factor), rdf.loc[mask, 'Cote'] * 2)))
        rdf['mrb'] = rdf['mrb'].astype(int)
        return rdf

    def select_squad(self, df_evaluated_players: pd.DataFrame, formation_key: str, target_squad_size: int) -> Tuple[pd.DataFrame, Dict]:
        squad_df = pd.DataFrame()
        available_players = df_evaluated_players.sort_values(by='pvs', ascending=False).copy()
        
        # Step 1: Define positional targets based on formation, minimums, and squad size
        pos_targets = self.squad_minimums.copy()
        flex_spots = target_squad_size - sum(pos_targets.values())
        if flex_spots > 0:
            non_gk_starters = sum(c for p, c in self.formations[formation_key].items() if p != 'GK')
            if non_gk_starters > 0:
                formation_ratios = {pos: count / non_gk_starters for pos, count in self.formations[formation_key].items() if pos != 'GK'}
                # Distribute flex spots proportionally based on formation
                for i in range(flex_spots):
                    pos_to_add = max(formation_ratios, key=lambda p: formation_ratios[p] * (1 - pos_targets[p] / target_squad_size))
                    pos_targets[pos_to_add] += 1

        # Step 2: Initial squad selection to meet targets with highest PVS players
        for pos, target_count in pos_targets.items():
            players_for_pos = available_players[available_players['simplified_position'] == pos].head(target_count)
            squad_df = pd.concat([squad_df, players_for_pos])
            available_players = available_players.drop(players_for_pos.index)
        
        # Step 3: Assign starters
        squad_df['is_starter'] = False
        for pos, count in self.formations[formation_key].items():
            starters_for_pos = squad_df[squad_df['simplified_position'] == pos].sort_values(by='pvs', ascending=False).head(count)
            squad_df.loc[starters_for_pos.index, 'is_starter'] = True

        # Step 4: Robust Budget Optimization
        current_mrb = squad_df['mrb'].sum()
        while current_mrb > self.budget:
            best_swap = None
            max_efficiency = -1
            
            # Find the most efficient downgrade possible across the whole team
            for _, p_old in squad_df[~squad_df['is_starter']].iterrows():
                replacements = available_players[(available_players['simplified_position'] == p_old['simplified_position']) & (available_players['mrb'] < p_old['mrb'])]
                if replacements.empty: continue
                p_new = replacements.iloc[0] # Best available replacement
                
                cost_saved = p_old['mrb'] - p_new['mrb']
                pvs_lost = p_old['pvs'] - p_new['pvs']
                efficiency = cost_saved / (pvs_lost + 0.1) # Add small epsilon to avoid division by zero
                
                if efficiency > max_efficiency:
                    max_efficiency = efficiency
                    best_swap = (p_old, p_new)

            if best_swap:
                p_old, p_new = best_swap
                squad_df = squad_df.drop(p_old.name)
                squad_df = pd.concat([squad_df, p_new.to_frame().T])
                available_players = available_players.drop(p_new.name)
                available_players = pd.concat([available_players, p_old.to_frame().T])
                current_mrb = squad_df['mrb'].sum()
            else:
                break # No possible downgrade found, exit loop

        summary = {'total_players': len(squad_df), 'total_cost': int(squad_df['mrb'].sum()), 'remaining_budget': int(self.budget - squad_df['mrb'].sum()), 'position_counts': squad_df['simplified_position'].value_counts().to_dict(), 'total_squad_pvs': round(squad_df['pvs'].sum(), 2), 'total_starters_pvs': round(squad_df[squad_df['is_starter']]['pvs'].sum(), 2)}
        return squad_df, summary

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
        if valid_ratings:
            goals = sum(r[1] for r in ratings_with_goals if r[0] is not None)
            kpi_data.append({'player_id': row['player_id'], KPI_PERFORMANCE: np.mean(valid_ratings), KPI_POTENTIAL: np.mean(sorted(valid_ratings, reverse=True)[:5]), KPI_REGULARITY: (len(valid_ratings) / len(gw_cols) * 100) if gw_cols else 0, KPI_GOALS: float(goals)})
    if not kpi_data: return pd.DataFrame()
    kpi_df = pd.DataFrame(kpi_data)
    for kpi in PLAYER_KPI_COLUMNS:
        max_val = kpi_df[kpi].max() if not kpi_df.empty else 0
        kpi_df[f"norm_{kpi}"] = (kpi_df[kpi] / max_val * 100) if max_val > 0 else 0
    return kpi_df

def main():
    st.markdown('<h1 class="main-header">🏆 MPG Hybrid Strategist v13.0</h1>', unsafe_allow_html=True)
    strategist = MPGAuctionStrategist()

    if 'current_profile_name' not in st.session_state:
        st.session_state.current_profile_name = "Balanced Value"
        st.session_state.kpi_weights = PREDEFINED_PROFILES["Balanced Value"]["kpi_weights"]
        st.session_state.mrb_params_per_pos = PREDEFINED_PROFILES["Balanced Value"]["mrb_params_per_pos"]
        st.session_state.team_rank_weight = PREDEFINED_PROFILES["Balanced Value"]["team_rank_weight"]
    if 'saved_squads' not in st.session_state: st.session_state.saved_squads = []

    st.sidebar.markdown('<h2 class="section-header" style="margin-top:0;">📁 File Inputs</h2>', unsafe_allow_html=True)
    hist_file = st.sidebar.file_uploader("1. Upload Historical Data", type=['csv', 'xlsx'])
    new_season_file = st.sidebar.file_uploader("2. Upload New Season Player List", type=['csv', 'xlsx'])

    if not hist_file or not new_season_file:
        st.info("👈 Please upload both files to begin."); return

    df_hist, df_new, returning_ids, new_player_ids = load_and_reconcile_players(hist_file, new_season_file)
    if df_hist is None: return
    
    df_returning_kpis = calculate_historical_kpis(df_hist, returning_ids)
    
    if 'team_tiers' not in st.session_state: st.session_state.team_tiers = {t: [] for t in ["Winner", "European", "Average", "Relegation"]}
    if 'new_player_kpis_df' not in st.session_state: st.session_state.new_player_kpis_df = pd.DataFrame()

    st.markdown('<h2 class="section-header">1. Team Ranking Setup</h2>', unsafe_allow_html=True)
    all_clubs = sorted(df_new['Club'].unique())
    tier_names = ["Winner", "European", "Average", "Relegation"]
    tier_cols = st.columns(len(tier_names))
    
    assigned_clubs = {club for tier_list in st.session_state.team_tiers.values() for club in tier_list}
    for i, tier in enumerate(tier_names):
        with tier_cols[i]:
            current_selection = [c for c in st.session_state.team_tiers.get(tier, []) if c in all_clubs]
            options_for_this_tier = sorted(list((set(all_clubs) - assigned_clubs) | set(current_selection)))
            st.session_state.team_tiers[tier] = st.multiselect(f"**{tier} Tier**", options=options_for_this_tier, default=current_selection, key=f"tier_{tier}")

    st.markdown('<hr><h2 class="section-header">2. New Player KPI Setup</h2>', unsafe_allow_html=True)
    df_new_players_info = df_new[df_new['player_id'].isin(new_player_ids)].copy()

    if df_new_players_info.empty: st.info("No new players identified.")
    else:
        if st.session_state.new_player_kpis_df.empty or set(st.session_state.new_player_kpis_df['player_id']) != set(df_new_players_info['player_id']):
            default_kpis = {}
            for pos in POSITIONS:
                default_kpis[pos] = {}
                returning_players_in_pos_ids = df_new[(df_new['simplified_position'] == pos) & (df_new['player_id'].isin(returning_ids))]['player_id']
                df_returning_pos = df_returning_kpis[df_returning_kpis['player_id'].isin(returning_players_in_pos_ids)]
                for kpi in PLAYER_KPI_COLUMNS:
                    default_kpis[pos][kpi] = df_returning_pos[kpi].median() if not df_returning_pos.empty and df_returning_pos[kpi].notna().any() else 50.0
            new_player_data = [{'player_id': p['player_id'], 'Joueur': p['Joueur'], 'Club': p['Club'], 'Poste': p['Poste'], **{kpi: default_kpis.get(p['simplified_position'], {}).get(kpi, 50.0) for kpi in PLAYER_KPI_COLUMNS}} for _, p in df_new_players_info.iterrows()]
            st.session_state.new_player_kpis_df = pd.DataFrame(new_player_data)
        st.markdown(f"Define KPIs for **{len(df_new_players_info)}** new players using the table below (scores are 0-100).")
        st.session_state.new_player_kpis_df = st.data_editor(st.session_state.new_player_kpis_df, column_config={"player_id": None, "Joueur": st.column_config.TextColumn(disabled=True), "Club": st.column_config.TextColumn(disabled=True), "Poste": st.column_config.TextColumn(disabled=True), **{kpi: st.column_config.NumberColumn(f"{kpi.replace('Estimation','')}", min_value=0, max_value=100, step=1) for kpi in PLAYER_KPI_COLUMNS}}, hide_index=True, key="new_player_editor")

    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("### 3. PVS & Squad Parameters")
    
    def apply_profile():
        profile_name = st.session_state.profile_selector
        if profile_name != "Custom" and profile_name in PREDEFINED_PROFILES:
            profile = PREDEFINED_PROFILES[profile_name]
            st.session_state.team_rank_weight = profile["team_rank_weight"]
            st.session_state.kpi_weights = profile["kpi_weights"]
            st.session_state.mrb_params_per_pos = profile["mrb_params_per_pos"]
        st.session_state.current_profile_name = profile_name

    st.sidebar.selectbox("Select Profile", options=list(PREDEFINED_PROFILES.keys()), key="profile_selector", on_change=apply_profile)
    trw_ui = st.sidebar.slider("Team Ranking KPI Weight", 0.0, 1.0, st.session_state.team_rank_weight, 0.05, key="trw_slider")
    if trw_ui != st.session_state.team_rank_weight:
        st.session_state.current_profile_name = "Custom"; st.session_state.team_rank_weight = trw_ui
    
    st.session_state.formation_key = st.sidebar.selectbox("Formation", list(strategist.formations.keys()), key="formation_selector")
    st.session_state.squad_size = st.sidebar.number_input("Squad Size", min_value=18, max_value=30, value=DEFAULT_SQUAD_SIZE, key="squad_size_selector")

    if st.sidebar.button("🚀 Generate Optimal Squad", type="primary"):
        df_new_kpis_from_editor = st.session_state.new_player_kpis_df.copy()
        for kpi in PLAYER_KPI_COLUMNS: df_new_kpis_from_editor[f"norm_{kpi}"] = df_new_kpis_from_editor[kpi]
        
        kpi_cols_to_keep = ['player_id'] + PLAYER_KPI_COLUMNS + [f'norm_{kpi}' for kpi in PLAYER_KPI_COLUMNS]
        df_new_kpis_clean = df_new_kpis_from_editor[kpi_cols_to_keep]
        all_kpis_df = pd.concat([df_returning_kpis, df_new_kpis_clean], ignore_index=True)

        tier_map = {100: "Winner", 75: "European", 50: "Average", 25: "Relegation"}
        club_to_score = {club: score for score, tier in tier_map.items() for club in st.session_state.team_tiers[tier]}
        df_new['Cote'] = pd.to_numeric(df_new['Cote'], errors='coerce').fillna(1)
        
        df_merged = pd.merge(df_new, all_kpis_df, on='player_id', how='left').dropna(subset=[f"norm_{kpi}" for kpi in PLAYER_KPI_COLUMNS])
        df_merged[KPI_TEAM_RANK] = df_merged['Club'].map(club_to_score).fillna(50)

        with st.spinner("🧠 Analyzing players and building your squad..."):
            df_pvs = strategist.calculate_pvs(df_merged, st.session_state.team_rank_weight, st.session_state.kpi_weights)
            st.session_state.df_full_eval = strategist.calculate_mrb(df_pvs, st.session_state.mrb_params_per_pos)
            squad_df, summary = strategist.select_squad(st.session_state.df_full_eval, st.session_state.formation_key, st.session_state.squad_size)
            st.session_state.squad_df_result, st.session_state.squad_summary_result = squad_df, summary

    if 'squad_df_result' in st.session_state and not st.session_state.squad_df_result.empty:
        st.markdown('<hr><h2 class="section-header">🏆 Final Results</h2>', unsafe_allow_html=True)
        
        cols_to_display_map = {'Joueur': 'Player', 'Club': 'Club', 'simplified_position': 'Pos', 'Cote': 'Cost', 'mrb': 'Bid', 'pvs': 'PVS', 'is_starter': 'Starter', KPI_PERFORMANCE: 'Perf', KPI_POTENTIAL: 'Pot', KPI_REGULARITY: 'Reg', KPI_GOALS: 'Goals'}
        
        tab1, tab2 = st.tabs(["Optimal Squad", "Full Player Database"])
        
        with tab1:
            col_main, col_summary = st.columns([3, 1])
            with col_main:
                squad_display_df = st.session_state.squad_df_result.rename(columns=cols_to_display_map)
                final_squad_cols = [col for col in cols_to_display_map.values() if col in squad_display_df.columns]
                st.dataframe(squad_display_df[final_squad_cols], use_container_width=True, hide_index=True, column_config={"PVS": st.column_config.ProgressColumn("PVS", min_value=0, max_value=100, format="%.1f")})
            with col_summary:
                summary = st.session_state.squad_summary_result
                st.metric("Budget Spent (MRB)", f"€ {summary.get('total_cost', 0):.0f}", help=f"Remaining: € {summary.get('remaining_budget', 0):.0f}")
                st.metric("Squad Size", f"{summary.get('total_players', 0)} (Target: {st.session_state.squad_size})")
                st.metric("Total Squad PVS", f"{summary.get('total_squad_pvs', 0):.2f}")
                st.metric("Starters PVS", f"{summary.get('total_starters_pvs', 0):.2f}")
                
                st.markdown("---")
                st.markdown("#### 💾 Save Current Squad")
                squad_name = st.text_input("Enter a name for this squad:", key="squad_name_input")
                if st.button("Save Squad"):
                    if squad_name and not any(s['name'] == squad_name for s in st.session_state.saved_squads):
                        st.session_state.saved_squads.append({'name': squad_name, 'summary': summary, 'squad_df': st.session_state.squad_df_result})
                        st.success(f"Squad '{squad_name}' saved!")
                        st.session_state.squad_name_input = "" 
                    elif not squad_name: st.warning("Please enter a name.")
                    else: st.warning(f"A squad named '{squad_name}' already exists.")
        
        with tab2:
            st.info("Search, sort, and analyze all evaluated players.")
            full_display_df = st.session_state.df_full_eval.rename(columns=cols_to_display_map)
            final_full_cols = [col for col in cols_to_display_map.values() if col in full_display_df.columns]
            st.dataframe(full_display_df[final_full_cols].sort_values(by='PVS', ascending=False), use_container_width=True, height=600, column_config={"PVS": st.column_config.ProgressColumn("PVS", min_value=0, max_value=100, format="%.1f")})
            
        if st.session_state.saved_squads:
            st.markdown('<hr><h2 class="section-header">💾 My Saved Squads</h2>', unsafe_allow_html=True)
            for i, saved in enumerate(reversed(st.session_state.saved_squads)):
                original_index = len(st.session_state.saved_squads) - 1 - i
                with st.expander(f"**{saved['name']}** | Cost: €{saved['summary']['total_cost']:.0f} | PVS: {saved['summary']['total_squad_pvs']:.2f}"):
                    saved_display_df = saved['squad_df'].rename(columns=cols_to_display_map)
                    final_saved_cols = [col for col in cols_to_display_map.values() if col in saved_display_df.columns]
                    st.dataframe(saved_display_df[final_saved_cols], use_container_width=True, hide_index=True)
                    if st.button("Delete", key=f"delete_{original_index}", type="primary"):
                        st.session_state.saved_squads.pop(original_index)
                        st.rerun()

if __name__ == "__main__":
    main()
