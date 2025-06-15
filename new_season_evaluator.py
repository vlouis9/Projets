import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Set
from io import BytesIO

# --- Page Configuration & Styling ---
st.set_page_config(
    page_title="MPG Hybrid Strategist v3",
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

# Profiles adapted for the new 5-KPI system
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
    """
    Houses all the core logic for player evaluation and squad selection.
    This version combines logic from both the historical and new season apps.
    """
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
                cote, pvs_player = int(row['Cote']), float(row['pvs'])
                bonus_factor = (pvs_player / 100.0) * max_prop_bonus
                return int(round(max(float(cote), min(cote * (1 + bonus_factor), float(cote * 2)))))
            rdf.loc[mask, 'mrb'] = rdf.loc[mask].apply(_calc_mrb_player, axis=1)
        rdf['mrb'] = rdf['mrb'].astype(int)
        rdf['value_per_cost'] = rdf['pvs'] / rdf['mrb'].replace(0, 1).astype(float)
        return rdf

    def select_squad(self, df_evaluated_players: pd.DataFrame, formation_key: str, target_squad_size: int) -> Tuple[pd.DataFrame, Dict]:
        # [span_0](start_span)This is the robust squad selection logic from season_with_data.txt[span_0](end_span)
        eligible_df = df_evaluated_players.drop_duplicates(subset=['player_id']).copy()
        eligible_df['mrb'] = eligible_df['mrb'].astype(int)
        squad = []

        def get_squad_ids(): return {p['player_id'] for p in squad}
        def get_pos_counts():
            counts = {pos: 0 for pos in self.squad_minimums}
            for p in squad: counts[p['pos']] += 1
            return counts

        def add_player(p_row, is_starter):
            if p_row['player_id'] in get_squad_ids(): return False
            if p_row['simplified_position'] == 'GK' and get_pos_counts().get('GK', 0) >= 2: return False
            squad.append({'player_id': p_row['player_id'], 'mrb': int(p_row['mrb']), 'pvs': float(p_row['pvs']), 'pos': p_row['simplified_position'], 'is_starter': is_starter})
            return True

        def remove_player(p_id):
            nonlocal squad
            squad = [p for p in squad if p['player_id'] != p_id]

        # Phase A1: Fill starters
        starters_map = self.formations[formation_key].copy()
        for _, row in eligible_df.sort_values(by='pvs', ascending=False).iterrows():
            if starters_map.get(row['simplified_position'], 0) > 0:
                if add_player(row, True):
                    starters_map[row['simplified_position']] -= 1

        # Phase A2: Meet position minimums
        for pos, min_needed in self.squad_minimums.items():
            while get_pos_counts().get(pos, 0) < min_needed:
                candidate = eligible_df[(eligible_df['simplified_position'] == pos) & (~eligible_df['player_id'].isin(get_squad_ids()))].sort_values(by='pvs', ascending=False).head(1)
                if candidate.empty or not add_player(candidate.iloc[0], False): break
        
        # Phase A3: Fill to target squad size
        while len(squad) < target_squad_size:
            candidate = eligible_df[~eligible_df['player_id'].isin(get_squad_ids())].sort_values(by='pvs', ascending=False).head(1)
            if candidate.empty or not add_player(candidate.iloc[0], False): break

        # Phase B & C: Budget Optimization
        current_mrb = sum(p['mrb'] for p in squad)
        for _ in range(target_squad_size * 2): # Limit iterations
            if current_mrb <= self.budget: break
            # Downgrade logic...
            best_downgrade = None
            for p_old in sorted(squad, key=lambda x: x['mrb'], reverse=True):
                replacements = eligible_df[(eligible_df['simplified_position'] == p_old['pos']) & (~eligible_df['player_id'].isin(get_squad_ids() | {p_old['player_id']})) & (eligible_df['mrb'] < p_old['mrb'])]
                if replacements.empty: continue
                p_new = replacements.iloc[0] # Simplification: take best PVS available at lower cost
                score = (p_old['mrb'] - p_new['mrb']) - (p_old['pvs'] - p_new['pvs']) # Heuristic
                if best_downgrade is None or score > best_downgrade[2]:
                    best_downgrade = (p_old, p_new.to_dict(), score)
            if best_downgrade:
                old, new_dict, _ = best_downgrade
                remove_player(old['player_id'])
                add_player(pd.Series(new_dict), old['is_starter'])
                current_mrb = sum(p['mrb'] for p in squad)
            else: break
        
        # Final formatting
        if not squad: return pd.DataFrame(), {}
        final_df = eligible_df[eligible_df['player_id'].isin(get_squad_ids())].copy()
        details_df = pd.DataFrame(squad).rename(columns={'mrb': 'mrb_actual_cost', 'pvs':'pvs_in_squad'})
        final_df = pd.merge(final_df, details_df, on='player_id')
        summary = {'total_players': len(final_df), 'total_cost': int(final_df['mrb_actual_cost'].sum()),'remaining_budget': int(self.budget - final_df['mrb_actual_cost'].sum()), 'position_counts': final_df['simplified_position'].value_counts().to_dict(),'total_squad_pvs': round(final_df['pvs_in_squad'].sum(), 2),'total_starters_pvs': round(final_df[final_df['is_starter']]['pvs_in_squad'].sum(), 2)}
        return final_df.sort_values(by=['is_starter', 'pvs_in_squad'], ascending=False), summary


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
    gw_cols = [col for col in df.columns if str(col).startswith('D')]
    kpi_data = []
    for _, row in df.iterrows():
        ratings, goals = [res for res in (extract_rating(row.get(gw)) for gw in gw_cols) if res[0] is not None], 0
        if ratings:
            just_ratings = [r[0] for r in ratings]
            goals = sum(r[1] for r in ratings)
            kpi_data.append({
                'player_id': row['player_id'],
                KPI_PERFORMANCE: np.mean(just_ratings),
                KPI_POTENTIAL: np.mean(sorted(just_ratings, reverse=True)[:5]),
                KPI_REGULARITY: (len(just_ratings) / len(gw_cols) * 100) if gw_cols else 0,
                KPI_GOALS: goals
            })
    kpi_df = pd.DataFrame(kpi_data)
    for kpi in PLAYER_KPI_COLUMNS:
        max_val = kpi_df[kpi].max() if not kpi_df.empty else 0
        kpi_df[f"norm_{kpi}"] = (kpi_df[kpi] / max_val * 100) if max_val > 0 else 0
    return kpi_df

# --- Main Streamlit App UI ---
def main():
    st.markdown('<h1 class="main-header">🏆 MPG Hybrid Strategist v3</h1>', unsafe_allow_html=True)
    strategist = MPGAuctionStrategist()

    # --- File Uploads ---
    st.sidebar.markdown('<h2 class="section-header" style="margin-top:0;">📁 File Inputs</h2>', unsafe_allow_html=True)
    hist_file = st.sidebar.file_uploader("1. Upload Historical Data", type=['csv', 'xlsx'])
    new_season_file = st.sidebar.file_uploader("2. Upload New Season Player List", type=['csv', 'xlsx'])

    if not hist_file or not new_season_file:
        st.info("👈 Please upload both files to begin."); return

    df_hist, df_new, returning_ids, new_player_ids = load_and_reconcile_players(hist_file, new_season_file)
    if df_hist is None: return
    
    df_returning_kpis = calculate_historical_kpis(df_hist, returning_ids)
    
    # Initialize session state for UI elements
    if 'team_tiers' not in st.session_state: st.session_state.team_tiers = {t: [] for t in ["Winner", "European", "Average", "Relegation"]}
    if 'new_player_kpis_df' not in st.session_state: st.session_state.new_player_kpis_df = pd.DataFrame()

    # --- UI Section 1: Team Ranking ---
    st.markdown('<h2 class="section-header">1. Team Ranking Setup</h2>', unsafe_allow_html=True)
    all_clubs = sorted(df_new['Club'].unique())
    tier_names = ["Winner", "European", "Average", "Relegation"]
    tier_cols = st.columns(len(tier_names))
    
    # Corrected Team Tier selection logic
    assigned_clubs = {club for tier_list in st.session_state.team_tiers.values() for club in tier_list}
    for i, tier in enumerate(tier_names):
        with tier_cols[i]:
            current_selection = st.session_state.team_tiers[tier]
            available_for_tier = sorted(list(set(all_clubs) - assigned_clubs) | set(current_selection))
            st.session_state.team_tiers[tier] = st.multiselect(f"**{tier} Tier**", options=available_for_tier, default=current_selection)

    # --- UI Section 2: New Player KPI Input (Improved with st.data_editor) ---
    st.markdown('<hr><h2 class="section-header">2. New Player KPI Setup</h2>', unsafe_allow_html=True)
    df_new_players_info = df_new[df_new['player_id'].isin(new_player_ids)].copy()

    if df_new_players_info.empty:
        st.info("No new players identified.")
    else:
        # Prepare the dataframe for the data editor
        if st.session_state.new_player_kpis_df.empty:
            kpi_ranges, default_kpis = {}, {}
            for pos in ['GK', 'DEF', 'MID', 'FWD']:
                kpi_ranges[pos], default_kpis[pos] = {}, {}
                pos_players_hist = df_new[df_new['simplified_position'] == pos]['player_id'].isin(returning_ids)
                df_returning_pos = df_returning_kpis[df_returning_kpis['player_id'].isin(df_new[pos_players_hist]['player_id'])]
                for kpi in PLAYER_KPI_COLUMNS:
                    norm_kpi = f"norm_{kpi}"
                    if not df_returning_pos.empty and norm_kpi in df_returning_pos.columns and df_returning_pos[norm_kpi].notna().any():
                        default_kpis[pos][kpi] = df_returning_pos[norm_kpi].median()
                    else: default_kpis[pos][kpi] = 50.0 # Fallback default
            
            new_player_data = []
            for _, player in df_new_players_info.iterrows():
                pos = player['simplified_position']
                player_data = {'Joueur': player['Joueur'], 'Club': player['Club'], 'Poste': player['Poste'], 'player_id': player['player_id']}
                for kpi in PLAYER_KPI_COLUMNS:
                    player_data[kpi] = default_kpis.get(pos, {}).get(kpi, 50.0)
                new_player_data.append(player_data)
            st.session_state.new_player_kpis_df = pd.DataFrame(new_player_data)

        st.markdown(f"Define KPIs for **{len(df_new_players_info)}** new players using the table below. Scores should be 0-100.")
        edited_df = st.data_editor(
            st.session_state.new_player_kpis_df,
            column_config={
                "Joueur": st.column_config.TextColumn("Player", disabled=True),
                "Club": st.column_config.TextColumn(disabled=True), "Poste": st.column_config.TextColumn(disabled=True),
                KPI_PERFORMANCE: st.column_config.NumberColumn("Perf.", min_value=0, max_value=100, step=1),
                KPI_POTENTIAL: st.column_config.NumberColumn("Pot.", min_value=0, max_value=100, step=1),
                KPI_REGULARITY: st.column_config.NumberColumn("Reg.", min_value=0, max_value=100, step=1),
                KPI_GOALS: st.column_config.NumberColumn("Goals", min_value=0, max_value=100, step=1),
            },
            hide_index=True, key="new_player_editor"
        )
        st.session_state.new_player_kpis_df = edited_df

    # --- UI Section 3: Final Calculation Button in Sidebar ---
    st.sidebar.markdown("<hr>", unsafe_allow_html=True)
    st.sidebar.markdown("### 3. PVS & Squad Parameters")
    st.session_state.team_rank_weight = st.sidebar.slider("Team Ranking KPI Weight", 0.0, 1.0, 0.2, 0.05)
    st.session_state.formation_key = st.sidebar.selectbox("Formation", list(strategist.formations.keys()))
    st.session_state.squad_size = st.sidebar.number_input("Squad Size", min_value=18, max_value=30, value=DEFAULT_SQUAD_SIZE)

    if st.sidebar.button("🚀 Generate Optimal Squad", type="primary"):
        # 1. Consolidate KPIs
        df_new_kpis = st.session_state.new_player_kpis_df.copy()
        for kpi in PLAYER_KPI_COLUMNS: # Normalize the edited data
            df_new_kpis[f"norm_{kpi}"] = df_new_kpis[kpi]
        
        df_all_kpis = pd.concat([df_returning_kpis, df_new_kpis], ignore_index=True)

        # 2. Add Team Ranking Score
        tier_map = {100: "Winner", 75: "European", 50: "Average", 25: "Relegation"}
        club_to_score = {club: score for score, tier in tier_map.items() for club in st.session_state.team_tiers[tier]}
        
        df_new['Cote'] = pd.to_numeric(df_new['Cote'], errors='coerce').fillna(1)
        df_merged = pd.merge(df_new, df_all_kpis, on='player_id', how='left')
        df_merged[KPI_TEAM_RANK] = df_merged['Club'].map(club_to_score).fillna(50)

        # 3. Calculate Final PVS, MRB, and Select Squad
        with st.spinner("🧠 Analyzing players and building your squad..."):
            # Using placeholder weights for now, full implementation would use sidebar controls
            weights = PREDEFINED_PROFILES["Balanced Value"]["kpi_weights"]
            mrb_params = PREDEFINED_PROFILES["Balanced Value"]["mrb_params_per_pos"]

            df_pvs = strategist.calculate_pvs(df_merged, st.session_state.team_rank_weight, weights)
            st.session_state.df_full_eval = strategist.calculate_mrb(df_pvs, mrb_params)
            
            squad_df, summary = strategist.select_squad(st.session_state.df_full_eval, st.session_state.formation_key, st.session_state.squad_size)
            st.session_state.squad_df_result = squad_df
            st.session_state.squad_summary_result = summary
    
    # --- Display Results in Tabs ---
    if 'squad_df_result' in st.session_state:
        st.markdown('<hr><h2 class="section-header">🏆 Final Results</h2>', unsafe_allow_html=True)
        tab1, tab2 = st.tabs(["Optimal Squad", "Full Player Database"])
        
        with tab1:
            st.dataframe(st.session_state.squad_df_result)
            st.json(st.session_state.squad_summary_result)

        with tab2:
            st.info("Search, sort, and analyze all evaluated players.")
            st.dataframe(st.session_state.df_full_eval, use_container_width=True, height=600)

if __name__ == "__main__":
    main()
