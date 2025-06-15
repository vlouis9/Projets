import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional, Set 
from pandas.api.types import CategoricalDtype

# Page configuration
st.set_page_config(
    page_title="MPG Ultimate Strategist", 
    page_icon="🚀",
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
    .tier-box {padding: 1rem; border-radius: 0.5rem; margin-bottom: 1rem;}
    .tier-winner {background-color: #ffd700; border: 2px solid #c0a000;}
    .tier-europe {background-color: #98fb98; border: 2px solid #7ccd7c;}
    .tier-average {background-color: #b0e0e6; border: 2px solid #87ceeb;}
    .tier-relegation {background-color: #f08080; border: 2px solid #cd5c5c;}
    .club-item {padding: 0.5rem; margin: 0.25rem 0; background-color: #f0f0f0; border-radius: 0.25rem;}
</style>
""", unsafe_allow_html=True)

# --- Subjective KPIs ---
KPI_PERFORMANCE = "PerformanceEstimation"
KPI_POTENTIAL = "PotentialEstimation"
KPI_REGULARITY = "RegularityEstimation"
KPI_GOALS = "GoalsEstimation"
KPI_TEAM_TIER = "TeamTierEstimation"
SUBJECTIVE_KPI_COLUMNS = [KPI_PERFORMANCE, KPI_POTENTIAL, KPI_REGULARITY, KPI_GOALS, KPI_TEAM_TIER]

# Constants
DEFAULT_SQUAD_SIZE = 20
DEFAULT_FORMATION = "4-4-2"
TIER_VALUES = {
    "Winner": 100,
    "European": 75,
    "Average": 50,
    "Relegation": 25
}

# --- Predefined Profiles ---
PREDEFINED_PROFILES = {
    "Custom": "custom",
    "Balanced Value": {
        "kpi_weights": {
            'GK':  {KPI_PERFORMANCE: 0.40, KPI_POTENTIAL: 0.10, KPI_REGULARITY: 0.40, KPI_GOALS: 0.00, KPI_TEAM_TIER: 0.10},
            'DEF': {KPI_PERFORMANCE: 0.35, KPI_POTENTIAL: 0.15, KPI_REGULARITY: 0.30, KPI_GOALS: 0.10, KPI_TEAM_TIER: 0.10},
            'MID': {KPI_PERFORMANCE: 0.30, KPI_POTENTIAL: 0.20, KPI_REGULARITY: 0.20, KPI_GOALS: 0.20, KPI_TEAM_TIER: 0.10},
            'FWD': {KPI_PERFORMANCE: 0.25, KPI_POTENTIAL: 0.20, KPI_REGULARITY: 0.15, KPI_GOALS: 0.30, KPI_TEAM_TIER: 0.10}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.3},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.4},
            'MID': {'max_proportional_bonus_at_pvs100': 0.6}, 
            'FWD': {'max_proportional_bonus_at_pvs100': 0.8}
        }
    },
    "Focus on High Potential": {
        "kpi_weights": {
            'GK':  {KPI_PERFORMANCE: 0.15, KPI_POTENTIAL: 0.50, KPI_REGULARITY: 0.25, KPI_GOALS: 0.00, KPI_TEAM_TIER: 0.10},
            'DEF': {KPI_PERFORMANCE: 0.15, KPI_POTENTIAL: 0.50, KPI_REGULARITY: 0.15, KPI_GOALS: 0.10, KPI_TEAM_TIER: 0.10},
            'MID': {KPI_PERFORMANCE: 0.10, KPI_POTENTIAL: 0.50, KPI_REGULARITY: 0.10, KPI_GOALS: 0.20, KPI_TEAM_TIER: 0.10},
            'FWD': {KPI_PERFORMANCE: 0.10, KPI_POTENTIAL: 0.45, KPI_REGULARITY: 0.05, KPI_GOALS: 0.30, KPI_TEAM_TIER: 0.10}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.5},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.6},
            'MID': {'max_proportional_bonus_at_pvs100': 0.8}, 
            'FWD': {'max_proportional_bonus_at_pvs100': 1.0}
        }
    },
    "Emphasis on Performance & Regularity": {
        "kpi_weights": {
            'GK':  {KPI_PERFORMANCE: 0.45, KPI_POTENTIAL: 0.05, KPI_REGULARITY: 0.40, KPI_GOALS: 0.00, KPI_TEAM_TIER: 0.10},
            'DEF': {KPI_PERFORMANCE: 0.40, KPI_POTENTIAL: 0.05, KPI_REGULARITY: 0.35, KPI_GOALS: 0.10, KPI_TEAM_TIER: 0.10},
            'MID': {KPI_PERFORMANCE: 0.35, KPI_POTENTIAL: 0.10, KPI_REGULARITY: 0.25, KPI_GOALS: 0.20, KPI_TEAM_TIER: 0.10},
            'FWD': {KPI_PERFORMANCE: 0.30, KPI_POTENTIAL: 0.10, KPI_REGULARITY: 0.20, KPI_GOALS: 0.30, KPI_TEAM_TIER: 0.10}
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

    @staticmethod 
    def simplify_position(position: str) -> str: 
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
        gw_cols_data = [{'name': col, 'number': int(match.group(1))} for col in df_columns 
                        if (match := re.fullmatch(r'D(\d+)', col))]
        return [col['name'] for col in sorted(gw_cols_data, key=lambda x: x['number'])]
    
    @staticmethod 
    def calculate_kpis(df: pd.DataFrame) -> pd.DataFrame: 
        rdf = df.copy() 
        all_gws = MPGAuctionStrategist.get_gameweek_columns(df.columns)
        
        # Initialize new columns
        rdf['season_avg_rating'] = 0.0
        rdf['potential_est'] = 0.0
        rdf['regularity_est'] = 0
        rdf['goals_est'] = 0
        rdf['games_played'] = 0

        for idx, row in rdf.iterrows():
            ratings = []
            goals_total = 0
            games_played = 0
            
            for gw_col in all_gws: 
                r, g, played, _ = MPGAuctionStrategist.extract_rating_goals_starter(row.get(gw_col))
                if played and r is not None:
                    ratings.append(r)
                    goals_total += g
                    games_played += 1
            
            # Performance Estimation (season average)
            if ratings:
                rdf.at[idx, 'season_avg_rating'] = np.mean(ratings)
                
                # Potential Estimation (average of top 5 ratings)
                top_ratings = sorted(ratings, reverse=True)[:5]
                rdf.at[idx, 'potential_est'] = np.mean(top_ratings)
            
            # Regularity Estimation (games played percentage)
            rdf.at[idx, 'regularity_est'] = (games_played / len(all_gws)) * 100 if all_gws else 0
            
            # Goals Estimation
            rdf.at[idx, 'goals_est'] = goals_total
            rdf.at[idx, 'games_played'] = games_played

        return rdf

    @staticmethod 
    def normalize_historical_kpis(df: pd.DataFrame) -> pd.DataFrame: 
        rdf = df.copy()
        
        # Normalize performance and potential to 0-100 scale
        rdf['norm_performance'] = rdf['season_avg_rating'] * 10  # 0-10 -> 0-100
        rdf['norm_potential'] = rdf['potential_est'] * 10  # 0-10 -> 0-100
        
        # Regularity is already in percentage (0-100)
        rdf['norm_regularity'] = rdf['regularity_est']
        
        # Normalize goals by position
        for pos in ['DEF', 'MID', 'FWD']:
            mask = rdf['simplified_position'] == pos
            if mask.any():
                max_goals = rdf.loc[mask, 'goals_est'].max()
                if max_goals > 0:
                    rdf.loc[mask, 'norm_goals'] = (rdf.loc[mask, 'goals_est'] / max_goals) * 100
                else:
                    rdf.loc[mask, 'norm_goals'] = 0
        # For GK and unknown positions
        rdf['norm_goals'] = rdf.get('norm_goals', 0)
        
        # Clip all values to 0-100
        for col in ['norm_performance', 'norm_potential', 'norm_regularity', 'norm_goals']:
            rdf[col] = rdf[col].clip(0, 100)
            
        return rdf

    @staticmethod 
    def normalize_subjective_kpis(df: pd.DataFrame) -> pd.DataFrame: 
        rdf = df.copy()
        for kpi_col in [KPI_PERFORMANCE, KPI_POTENTIAL, KPI_REGULARITY, KPI_GOALS]:
            if kpi_col in rdf.columns:
                rdf[kpi_col] = rdf[kpi_col].clip(0, 100)
        return rdf

    @staticmethod 
    def calculate_pvs(df: pd.DataFrame, weights: Dict[str, Dict[str, float]]) -> pd.DataFrame: 
        rdf = df.copy()
        rdf['pvs'] = 0.0
        
        for pos, pos_weights in weights.items():
            mask = rdf['simplified_position'] == pos
            if not mask.any(): continue
            
            pvs_sum = pd.Series(0.0, index=rdf.loc[mask].index)
            total_weight = sum(pos_weights.values())
            
            for kpi_col in [KPI_PERFORMANCE, KPI_POTENTIAL, KPI_REGULARITY, KPI_GOALS, KPI_TEAM_TIER]:
                weight = pos_weights.get(kpi_col, 0)
                if kpi_col in rdf.columns:
                    pvs_sum += rdf.loc[mask, kpi_col] * weight
            
            if total_weight > 0:
                rdf.loc[mask, 'pvs'] = (pvs_sum / total_weight).clip(0, 100)
        
        return rdf

    @staticmethod 
    def calculate_mrb(df: pd.DataFrame, mrb_params_per_pos: Dict[str, Dict[str, float]]) -> pd.DataFrame: 
        rdf = df.copy()
        rdf['mrb'] = rdf['Cote'] 
        
        for pos, params in mrb_params_per_pos.items():
            mask = rdf['simplified_position'] == pos 
            if not mask.any(): continue
            max_bonus = params.get('max_proportional_bonus_at_pvs100', 0.5)

            def _calc_mrb(row):
                cote = row['Cote']
                pvs = row['pvs']
                bonus_factor = (pvs / 100) * max_bonus
                mrb_float = cote * (1 + bonus_factor)
                return min(max(cote, mrb_float), cote * 2)

            rdf.loc[mask, 'mrb'] = rdf.loc[mask].apply(_calc_mrb, axis=1)
        
        rdf['mrb'] = rdf['mrb'].astype(int)
        rdf['value_per_cost'] = rdf['pvs'] / rdf['mrb'].replace(0, 1)
        rdf['value_per_cost'].fillna(0, inplace=True)
        return rdf

    @staticmethod 
    def select_squad(df_evaluated_players: pd.DataFrame, formation_key: str, 
                    target_squad_size: int, budget: int) -> Tuple[pd.DataFrame, Dict]:
        # Implementation of squad selection algorithm
        # ... (similar to previous implementation but updated for new KPIs)
        # This would be a lengthy implementation - using placeholder logic
        formation = {
            "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
            # ... other formations
        }[formation_key]
        
        # Simplified squad selection logic
        df = df_evaluated_players.copy()
        df = df.sort_values('value_per_cost', ascending=False)
        
        squad = []
        position_counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
        total_cost = 0
        
        # Select starters
        for pos, count in formation.items():
            candidates = df[df['simplified_position'] == pos].head(count)
            for _, player in candidates.iterrows():
                if total_cost + player['mrb'] <= budget and len(squad) < 11:
                    squad.append(player)
                    position_counts[pos] += 1
                    total_cost += player['mrb']
                    df = df[df['player_id'] != player['player_id']]
        
        # Fill minimums
        min_requirements = {"GK": 2, "DEF": 6, "MID": 6, "FWD": 4}
        for pos, min_count in min_requirements.items():
            while position_counts[pos] < min_count and len(squad) < target_squad_size:
                candidates = df[df['simplified_position'] == pos]
                if not candidates.empty:
                    player = candidates.iloc[0]
                    if total_cost + player['mrb'] <= budget:
                        squad.append(player)
                        position_counts[pos] += 1
                        total_cost += player['mrb']
                        df = df[df['player_id'] != player['player_id']]
                    else:
                        break
                else:
                    break
        
        # Fill to squad size
        while len(squad) < target_squad_size and not df.empty:
            player = df.iloc[0]
            if total_cost + player['mrb'] <= budget:
                squad.append(player)
                total_cost += player['mrb']
                df = df[df['player_id'] != player['player_id']]
            else:
                break
        
        squad_df = pd.DataFrame(squad)
        summary = {
            'total_players': len(squad_df),
            'total_cost': total_cost,
            'remaining_budget': budget - total_cost,
            'position_counts': position_counts,
            'total_squad_pvs': squad_df['pvs'].sum(),
            'total_starters_pvs': squad_df.head(11)['pvs'].sum()
        }
        return squad_df, summary

# --- Data Loading Functions ---
@st.cache_data
def load_historical_data(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded_file)
        df['simplified_position'] = df['Poste'].apply(MPGAuctionStrategist.simplify_position)
        df['player_id'] = df.apply(MPGAuctionStrategist.create_player_id, axis=1)
        df = MPGAuctionStrategist.calculate_kpis(df)
        df = MPGAuctionStrategist.normalize_historical_kpis(df)
        return df
    except Exception as e:
        st.error(f"Error processing historical data: {e}")
        return None

@st.cache_data
def load_new_season_data(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        df = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded_file)
        df['simplified_position'] = df['Poste'].apply(MPGAuctionStrategist.simplify_position)
        df['player_id'] = df.apply(MPGAuctionStrategist.create_player_id, axis=1)
        df['Cote'] = pd.to_numeric(df['Cote'], errors='coerce').fillna(1).clip(lower=1).astype(int)
        return df
    except Exception as e:
        st.error(f"Error processing new season data: {e}")
        return None

def merge_player_data(historical_df, new_season_df):
    # Identify returning players
    merged = pd.merge(
        new_season_df, 
        historical_df[['player_id', 'norm_performance', 'norm_potential', 'norm_regularity', 'norm_goals']],
        on='player_id', 
        how='left'
    )
    
    # Identify new players
    new_players = merged[merged['norm_performance'].isna()]
    known_players = merged[~merged['norm_performance'].isna()]
    
    return known_players, new_players

def assign_team_tiers(clubs):
    tiers = {tier: [] for tier in TIER_VALUES}
    return {club: 50 for club in clubs}  # Default to average

# --- UI Components ---
def team_tier_ui(clubs):
    st.markdown("### 🏆 Team Tier Assignment")
    st.info("Drag clubs between tiers to assign values (Winner: 100, European: 75, Average: 50, Relegation: 25)")
    
    # Create columns for each tier
    cols = st.columns(4)
    tier_assignments = {tier: [] for tier in TIER_VALUES}
    
    with cols[0]:
        st.subheader("Winner Tier (100)")
        st.markdown('<div class="tier-box tier-winner">', unsafe_allow_html=True)
        tier_assignments["Winner"] = st.multiselect("", clubs, key="winner_tier", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with cols[1]:
        st.subheader("European Tier (75)")
        st.markdown('<div class="tier-box tier-europe">', unsafe_allow_html=True)
        tier_assignments["European"] = st.multiselect("", clubs, key="europe_tier", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with cols[2]:
        st.subheader("Average Tier (50)")
        st.markdown('<div class="tier-box tier-average">', unsafe_allow_html=True)
        tier_assignments["Average"] = st.multiselect("", clubs, key="average_tier", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with cols[3]:
        st.subheader("Relegation Tier (25)")
        st.markdown('<div class="tier-box tier-relegation">', unsafe_allow_html=True)
        tier_assignments["Relegation"] = st.multiselect("", clubs, key="relegation_tier", label_visibility="collapsed")
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Create club-tier mapping
    club_tiers = {}
    for tier, clubs_in_tier in tier_assignments.items():
        for club in clubs_in_tier:
            club_tiers[club] = TIER_VALUES[tier]
    
    # Set default for unassigned clubs
    all_clubs_set = set(clubs)
    assigned_clubs = set(club for clubs_in_tier in tier_assignments.values() for club in clubs_in_tier)
    for club in all_clubs_set - assigned_clubs:
        club_tiers[club] = 50  # Default to average
    
    return club_tiers

def new_player_kpi_ui(new_players_df, kpi_ranges):
    st.markdown("### 🆕 New Player KPI Assignment")
    st.info("Set KPIs for new players based on historical data ranges")
    
    kpis = [KPI_PERFORMANCE, KPI_POTENTIAL, KPI_REGULARITY, KPI_GOALS]
    options = [0, 25, 50, 75, 100]
    
    for idx, player in new_players_df.iterrows():
        with st.expander(f"{player['Joueur']} ({player['simplified_position']}, {player['Club']})"):
            cols = st.columns(4)
            for i, kpi in enumerate(kpis):
                pos = player['simplified_position']
                with cols[i]:
                    st.slider(
                        kpi.split("Estimation")[0], 
                        min_value=0, 
                        max_value=100, 
                        value=50, 
                        step=25,
                        key=f"{player['player_id']}_{kpi}"
                    )

# --- Main App ---
def main():
    st.markdown('<h1 class="main-header">🚀 MPG Ultimate Strategist</h1>', unsafe_allow_html=True)
    strategist = MPGAuctionStrategist()
    
    # Initialize session state
    if 'current_profile_name' not in st.session_state:
        st.session_state.current_profile_name = "Balanced Value"
        profile = PREDEFINED_PROFILES[st.session_state.current_profile_name]
        st.session_state.kpi_weights = profile["kpi_weights"]
        st.session_state.mrb_params = profile["mrb_params_per_pos"]
    
    # File uploaders
    st.sidebar.markdown("## 📁 Data Upload")
    historical_file = st.sidebar.file_uploader("Historical Data (CSV/Excel)", type=['csv', 'xlsx', 'xls'])
    new_season_file = st.sidebar.file_uploader("New Season Data (CSV/Excel)", type=['csv', 'xlsx', 'xls'])
    
    # Load data
    historical_df = load_historical_data(historical_file) if historical_file else None
    new_season_df = load_new_season_data(new_season_file) if new_season_file else None
    
    if historical_df is not None and new_season_df is not None:
        # Process data
        known_players, new_players = merge_player_data(historical_df, new_season_df)
        
        # Team tier assignment
        clubs = new_season_df['Club'].unique().tolist()
        club_tiers = team_tier_ui(clubs)
        
        # Add team tier to all players
        new_season_df[KPI_TEAM_TIER] = new_season_df['Club'].map(club_tiers)
        
        # New player KPI assignment
        if not new_players.empty:
            new_player_kpi_ui(new_players, None)  # Simplified version
        
        # Apply KPI values to new players
        for idx, player in new_players.iterrows():
            for kpi in [KPI_PERFORMANCE, KPI_POTENTIAL, KPI_REGULARITY, KPI_GOALS]:
                # In real implementation, get values from session state
                new_season_df.loc[new_season_df['player_id'] == player['player_id'], kpi] = 50  # Default
        
        # Profile configuration
        st.sidebar.markdown("## ⚙️ Configuration")
        profile_names = list(PREDEFINED_PROFILES.keys())
        selected_profile = st.sidebar.selectbox("Strategy Profile", profile_names, index=profile_names.index(st.session_state.current_profile_name))
        
        if selected_profile != st.session_state.current_profile_name:
            st.session_state.current_profile_name = selected_profile
            profile = PREDEFINED_PROFILES[selected_profile]
            st.session_state.kpi_weights = profile["kpi_weights"]
            st.session_state.mrb_params = profile["mrb_params_per_pos"]
        
        # KPI weights customization
        with st.sidebar.expander("📊 KPI Weights"):
            for pos in ['GK', 'DEF', 'MID', 'FWD']:
                st.subheader(pos)
                for kpi in SUBJECTIVE_KPI_COLUMNS:
                    current_weight = st.session_state.kpi_weights[pos].get(kpi, 0.0)
                    new_weight = st.slider(
                        f"{kpi.split('Estimation')[0]}", 
                        0.0, 1.0, current_weight, 0.05,
                        key=f"weight_{pos}_{kpi}"
                    )
                    st.session_state.kpi_weights[pos][kpi] = new_weight
        
        # Squad parameters
        st.sidebar.markdown("## 👥 Squad Parameters")
        formation = st.sidebar.selectbox("Formation", list(strategist.formations.keys()))
        squad_size = st.sidebar.number_input("Squad Size", 15, 30, DEFAULT_SQUAD_SIZE)
        
        # Calculate PVS and MRB
        if st.button("Calculate Player Values"):
            with st.spinner("Calculating PVS and MRB..."):
                # Prepare KPI columns
                for kpi in [KPI_PERFORMANCE, KPI_POTENTIAL, KPI_REGULARITY, KPI_GOALS]:
                    if kpi not in new_season_df.columns:
                        # Apply default values for known players
                        if kpi == KPI_PERFORMANCE:
                            new_season_df[kpi] = new_season_df.get('norm_performance', 50)
                        elif kpi == KPI_POTENTIAL:
                            new_season_df[kpi] = new_season_df.get('norm_potential', 50)
                        elif kpi == KPI_REGULARITY:
                            new_season_df[kpi] = new_season_df.get('norm_regularity', 50)
                        elif kpi == KPI_GOALS:
                            new_season_df[kpi] = new_season_df.get('norm_goals', 0)
                
                # Normalize
                df_normalized = MPGAuctionStrategist.normalize_subjective_kpis(new_season_df)
                
                # Calculate PVS
                df_pvs = MPGAuctionStrategist.calculate_pvs(
                    df_normalized, 
                    st.session_state.kpi_weights
                )
                
                # Calculate MRB
                df_mrb = MPGAuctionStrategist.calculate_mrb(
                    df_pvs, 
                    st.session_state.mrb_params
                )
                
                # Store results
                st.session_state.player_data = df_mrb
        
        # Squad selection
        if 'player_data' in st.session_state:
            if st.button("Build Optimal Squad"):
                with st.spinner("Selecting best squad..."):
                    squad_df, summary = strategist.select_squad(
                        st.session_state.player_data,
                        formation,
                        squad_size,
                        strategist.budget
                    )
                    st.session_state.squad_df = squad_df
                    st.session_state.squad_summary = summary
        
        # Display results
        if 'squad_df' in st.session_state:
            st.markdown("## 🏆 Recommended Squad")
            st.dataframe(st.session_state.squad_df[
                ['Joueur', 'Club', 'simplified_position', 'pvs', 'mrb', 'Cote']
            ].rename(columns={
                'Joueur': 'Player',
                'simplified_position': 'Position',
                'pvs': 'PVS',
                'mrb': 'MRB',
                'Cote': 'Base Price'
            }))
            
            st.markdown("### 📊 Squad Summary")
            summary = st.session_state.squad_summary
            st.metric("Total Cost", f"€{summary['total_cost']}")
            st.metric("Remaining Budget", f"€{summary['remaining_budget']}")
            st.metric("Total PVS", f"{summary['total_squad_pvs']:.1f}")
            st.metric("Starters PVS", f"{summary['total_starters_pvs']:.1f}")
            
            # Download button
            st.download_button(
                "💾 Download Squad",
                st.session_state.squad_df.to_csv(index=False),
                "mpg_optimal_squad.csv"
            )
    
    else:
        st.info("👋 Welcome to MPG Ultimate Strategist! Please upload both historical and new season data to begin.")
        st.image("https://images.unsplash.com/photo-1540747913346-19e32dc3e97e?auto=format&fit=crop&w=1200&q=80", 
                 caption="Football Strategy Dashboard")

if __name__ == "__main__":
    main()
