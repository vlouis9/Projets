import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="MPG Auction Strategist v3",
    page_icon="🚀", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem; font-weight: bold; color: #004080; 
        text-align: center; margin-bottom: 2rem; font-family: 'Roboto', sans-serif;
    }
    .section-header {
        font-size: 1.4rem; font-weight: bold; color: #006847; 
        margin-top: 1.5rem; margin-bottom: 1rem; border-bottom: 2px solid #006847; padding-bottom: 0.3rem;
    }
    .stButton>button {
        background-color: #004080; color: white; font-weight: bold;
        border-radius: 0.3rem; padding: 0.4rem 0.8rem; border: none; width: 100%;
    }
    .stButton>button:hover { background-color: #003060; color: white; }
    .stSlider [data-baseweb="slider"] { padding-bottom: 12px; }
    .css-1d391kg { background-color: #f8f9fa; padding-top: 1rem; } /* Sidebar class */
    .stExpander { border: 1px solid #e0e0e0; border-radius: 0.3rem; margin-bottom: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# --- Constants and Predefined Profiles ---
DEFAULT_N_RECENT_GAMES = 5
DEFAULT_MIN_RECENT_GAMES_PLAYED = 1
DEFAULT_SQUAD_SIZE = 20
DEFAULT_FORMATION = "4-4-2"

# New MRB Param per position: 'max_proportional_bonus_at_pvs100': Factor (0.0-1.0) for bonus if PVS=100.
# MRB = Cote * (1 + (PVS/100) * max_proportional_bonus_at_pvs100). Capped at 2*Cote.
PREDEFINED_PROFILES = {
    "Custom": "custom",
    "Balanced Value": {
        "n_recent_games": 5,
        "min_recent_games_played_filter": 1,
        "kpi_weights": {
            'GK':  {'recent_avg': 0.35, 'season_avg': 0.35, 'regularity': 0.30, 'recent_goals': 0.0, 'season_goals': 0.0},
            'DEF': {'recent_avg': 0.30, 'season_avg': 0.30, 'regularity': 0.40, 'recent_goals': 0.0, 'season_goals': 0.0},
            'MID': {'recent_avg': 0.25, 'season_avg': 0.25, 'regularity': 0.20, 'recent_goals': 0.15, 'season_goals': 0.15},
            'FWD': {'recent_avg': 0.20, 'season_avg': 0.20, 'regularity': 0.15, 'recent_goals': 0.25, 'season_goals': 0.20}
        },
        "mrb_params_per_pos": { # Max bonus if PVS=100. E.g., 0.5 means MRB can be Cote * 1.5 if PVS=100
            'GK':  {'max_proportional_bonus_at_pvs100': 0.3}, 
            'DEF': {'max_proportional_bonus_at_pvs100': 0.4}, 
            'MID': {'max_proportional_bonus_at_pvs100': 0.6}, 
            'FWD': {'max_proportional_bonus_at_pvs100': 0.8}  
        }
    },
    "Aggressive Bids (Pay for PVS)": {
        "n_recent_games": 5,
        "min_recent_games_played_filter": 0,
        "kpi_weights": { 
            'GK':  {'recent_avg': 0.4, 'season_avg': 0.4, 'regularity': 0.2, 'recent_goals': 0.0, 'season_goals': 0.0},
            'DEF': {'recent_avg': 0.35, 'season_avg': 0.35, 'regularity': 0.3, 'recent_goals': 0.0, 'season_goals': 0.0},
            'MID': {'recent_avg': 0.3, 'season_avg': 0.3, 'regularity': 0.1, 'recent_goals': 0.15, 'season_goals': 0.15},
            'FWD': {'recent_avg': 0.3, 'season_avg': 0.3, 'regularity': 0.05, 'recent_goals': 0.2, 'season_goals': 0.15}
        },
        "mrb_params_per_pos": { 
            'GK':  {'max_proportional_bonus_at_pvs100': 0.5}, 
            'DEF': {'max_proportional_bonus_at_pvs100': 0.6}, 
            'MID': {'max_proportional_bonus_at_pvs100': 0.8}, 
            'FWD': {'max_proportional_bonus_at_pvs100': 1.0} # Max bonus is 100% of Cote, so MRB up to 2x Cote
        }
    }
}

class MPGAuctionStrategist:
    def __init__(self):
        self.formations = { # Standard formations
            "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}, "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
            "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2}, "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
            "4-5-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1}, "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},
            "5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1}
        }
        self.squad_minimums = {"GK": 2, "DEF": 6, "MID": 6, "FWD": 4}
        self.budget = 500

    @property
    def squad_minimums_sum_val(self): return sum(self.squad_minimums.values())

    def simplify_position(self, position: str) -> str: # As before
        if pd.isna(position) or str(position).strip() == '': return 'UNKNOWN'
        pos = str(position).upper().strip()
        if pos == 'G': return 'GK'
        elif pos in ['D', 'DL', 'DC']: return 'DEF'
        elif pos in ['M', 'MD', 'MO']: return 'MID'
        elif pos == 'A': return 'FWD'
        else: return 'UNKNOWN'

    def create_player_id(self, row) -> str: # As before
        name = str(row.get('Joueur', '')).strip()
        simplified_pos = self.simplify_position(row.get('Poste', ''))
        club = str(row.get('Club', '')).strip()
        return f"{name}_{simplified_pos}_{club}"

    def extract_rating_and_goals(self, rating_str) -> Tuple[Optional[float], int, bool]: # As before
        if pd.isna(rating_str) or str(rating_str).strip() == '' or str(rating_str).strip() == '0': return None, 0, False
        rating_val_str = str(rating_str).strip()
        goals = rating_val_str.count('*')
        clean_rating_str = re.sub(r'[()\*]', '', rating_val_str)
        try: return float(clean_rating_str), goals, True
        except ValueError: return None, 0, False

    def get_gameweek_columns(self, df_columns: List[str]) -> List[str]: # As before
        gw_cols_data = [{'name': col, 'number': int(match.group(1))} for col in df_columns if (match := re.fullmatch(r'D(\d+)', col))]
        return [col['name'] for col in sorted(gw_cols_data, key=lambda x: x['number'])]

    def calculate_kpis(self, df: pd.DataFrame, n_recent: int) -> pd.DataFrame: # As before (Recent form based on played games in N window)
        rdf = df.copy()
        all_gws = self.get_gameweek_columns(df.columns)
        rdf[['recent_avg_rating', 'season_avg_rating']] = 0.0
        rdf[['recent_goals', 'season_goals', 'recent_games_played_count']] = 0

        for idx, row in rdf.iterrows():
            s_ratings, s_goals = [], 0
            for gw_col in all_gws:
                r, g, played = self.extract_rating_and_goals(row.get(gw_col))
                if played and r is not None: s_ratings.append(r); s_goals += g
            rdf.at[idx, 'season_avg_rating'] = np.mean(s_ratings) if s_ratings else 0.0
            rdf.at[idx, 'season_goals'] = s_goals

            rec_gws = all_gws[-n_recent:]
            rec_ratings, rec_goals_sum, rec_games_played = [], 0, 0
            for gw_col in rec_gws:
                r, g, played = self.extract_rating_and_goals(row.get(gw_col))
                if played and r is not None: rec_ratings.append(r); rec_goals_sum += g; rec_games_played += 1
            rdf.at[idx, 'recent_avg_rating'] = np.mean(rec_ratings) if rec_ratings else 0.0
            rdf.at[idx, 'recent_goals'] = rec_goals_sum
            rdf.at[idx, 'recent_games_played_count'] = rec_games_played
        
        # Ensure goals are integer
        rdf[['recent_goals', 'season_goals', 'recent_games_played_count']] = rdf[['recent_goals', 'season_goals', 'recent_games_played_count']].astype(int)
        return rdf

    def normalize_kpis(self, df: pd.DataFrame) -> pd.DataFrame: # As before
        rdf = df.copy()
        rdf['norm_recent_avg'] = np.clip(rdf['recent_avg_rating'] * 10, 0, 100)
        rdf['norm_season_avg'] = np.clip(rdf['season_avg_rating'] * 10, 0, 100)
        rdf['norm_regularity'] = pd.to_numeric(rdf['%Titu'], errors='coerce').fillna(0).clip(0, 100)
        rdf[['norm_recent_goals', 'norm_season_goals']] = 0.0
        for pos in ['MID', 'FWD']:
            mask = rdf['simplified_position'] == pos
            if mask.any():
                rdf.loc[mask, 'norm_recent_goals'] = np.clip(rdf.loc[mask, 'recent_goals'] * 20, 0, 100)
                max_sg = rdf.loc[mask, 'season_goals'].max()
                rdf.loc[mask, 'norm_season_goals'] = np.clip((rdf.loc[mask, 'season_goals'] / max_sg * 100) if max_sg > 0 else 0, 0, 100)
        return rdf

    def calculate_pvs(self, df: pd.DataFrame, weights: Dict[str, Dict[str, float]]) -> pd.DataFrame: # As before (PVS 0-100)
        rdf = df.copy()
        rdf['pvs'] = 0.0
        for pos, w in weights.items():
            mask = rdf['simplified_position'] == pos;
            if not mask.any(): continue
            pvs = pd.Series(0.0, index=rdf.loc[mask].index)
            pvs += rdf.loc[mask, 'norm_recent_avg'].fillna(0) * w.get('recent_avg',0)
            pvs += rdf.loc[mask, 'norm_season_avg'].fillna(0) * w.get('season_avg',0)
            pvs += rdf.loc[mask, 'norm_regularity'].fillna(0) * w.get('regularity',0)
            if pos in ['MID', 'FWD']:
                pvs += rdf.loc[mask, 'norm_recent_goals'].fillna(0) * w.get('recent_goals',0)
                pvs += rdf.loc[mask, 'norm_season_goals'].fillna(0) * w.get('season_goals',0)
            rdf.loc[mask, 'pvs'] = pvs.clip(0, 100)
        return rdf

    def calculate_mrb(self, df: pd.DataFrame, mrb_params_per_pos: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        rdf = df.copy()
        # Cote is already int from main processing
        rdf['mrb'] = rdf['Cote'] 

        for pos_simplified, params in mrb_params_per_pos.items():
            mask = rdf['simplified_position'] == pos_simplified
            if not mask.any(): continue

            max_prop_bonus = params.get('max_proportional_bonus_at_pvs100', 0.5) # Default to 0.5 if not found

            def _calc_mrb_player_v2(row):
                cote = int(row['Cote']) # Ensure integer
                pvs_player = float(row['pvs']) # PVS is 0-100
                
                pvs_scaled_0_1 = pvs_player / 100.0
                pvs_derived_bonus_factor = pvs_scaled_0_1 * max_prop_bonus
                
                mrb_float = cote * (1 + pvs_derived_bonus_factor)
                
                # Apply caps: Max 2x Cote, Min Cote
                mrb_capped = min(mrb_float, float(cote * 2))
                final_mrb = max(float(cote), mrb_capped)
                return int(round(final_mrb))

            rdf.loc[mask, 'mrb'] = rdf.loc[mask].apply(_calc_mrb_player_v2, axis=1)
        
        rdf['mrb'] = rdf['mrb'].astype(int) # Ensure final MRB is integer
        safe_mrb = rdf['mrb'].replace(0, np.nan).astype(float) # MRB should not be 0 due to max(cote, mrb)
        rdf['value_per_cost'] = rdf['pvs'] / safe_mrb
        rdf['value_per_cost'].fillna(0, inplace=True)
        return rdf

    def select_squad(self, df: pd.DataFrame, formation_key: str, target_squad_size: int,
                     min_recent_games_played: int) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        
        eligible_df = df.copy()
        if min_recent_games_played > 0:
            eligible_df = eligible_df[eligible_df['recent_games_played_count'] >= min_recent_games_played]
        if 'Indispo ?' in eligible_df.columns: eligible_df = eligible_df[~eligible_df['Indispo ?']]
        
        if eligible_df.empty: return pd.DataFrame(), {}

        eligible_df = eligible_df.drop_duplicates(subset=['player_id'])
        eligible_df['mrb'] = eligible_df['mrb'].astype(int) # Ensure MRB is int for budget

        selected_details = [] 
        current_budget = self.budget
        current_pos_counts = {pos: 0 for pos in ['GK', 'DEF', 'MID', 'FWD']}
        
        # --- Phase 1: Select Starters (Prioritize PVS) ---
        starters_needed = self.formations[formation_key].copy()
        for pos, num_needed in starters_needed.items():
            candidates = eligible_df[(eligible_df['simplified_position'] == pos) & 
                                     (~eligible_df['player_id'].isin([p['player_id'] for p in selected_details]))
                                    ].sort_values(by='pvs', ascending=False)
            added_count = 0
            for _, p_row in candidates.iterrows():
                if added_count >= num_needed: break
                if p_row['mrb'] <= current_budget:
                    selected_details.append({'player_id': p_row['player_id'], 'is_starter': True, 'mrb_cost': p_row['mrb'], 'pvs': p_row['pvs'], 'position': pos})
                    current_budget -= p_row['mrb']
                    current_pos_counts[pos] += 1
                    added_count += 1
        
        # --- Phase 2: Fulfill Overall Squad Minimums (Prioritize PVS, then PVS/MRB if needed) ---
        for pos, overall_min in self.squad_minimums.items():
            needed = max(0, overall_min - current_pos_counts[pos])
            if needed == 0: continue
            # Try PVS first for higher quality bench, then fall back to PVS/MRB if budget is an issue for top PVS.
            # For MVP, let's use PVS/MRB for efficiency in filling these minimums, user can tune weights for PVS.
            candidates = eligible_df[(eligible_df['simplified_position'] == pos) & 
                                     (~eligible_df['player_id'].isin([p['player_id'] for p in selected_details]))
                                    ].sort_values(by='value_per_cost', ascending=False) 
            added_count = 0
            for _, p_row in candidates.iterrows():
                if added_count >= needed: break
                if p_row['mrb'] <= current_budget:
                    selected_details.append({'player_id': p_row['player_id'], 'is_starter': False, 'mrb_cost': p_row['mrb'], 'pvs': p_row['pvs'], 'position': pos})
                    current_budget -= p_row['mrb']
                    current_pos_counts[pos] += 1
                    added_count += 1
        
        # --- Phase 3: Complete to Total Squad Size (PVS/MRB for remaining bench) ---
        slots_to_fill = max(0, target_squad_size - len(selected_details))
        if slots_to_fill > 0:
            candidates = eligible_df[(~eligible_df['player_id'].isin([p['player_id'] for p in selected_details]))
                                    ].sort_values(by='value_per_cost', ascending=False)
            added_count = 0
            for _, p_row in candidates.iterrows():
                if added_count >= slots_to_fill: break
                if p_row['mrb'] <= current_budget:
                    selected_details.append({'player_id': p_row['player_id'], 'is_starter': False, 'mrb_cost': p_row['mrb'], 'pvs': p_row['pvs'], 'position': p_row['simplified_position']})
                    current_budget -= p_row['mrb']
                    current_pos_counts[p_row['simplified_position']] += 1
                    added_count += 1

        # --- Phase 4: Iterative Budget Utilization / Upgrade (MVP - Simple Pass) ---
        # Attempt to use remaining budget to upgrade PVS of bench players
        if current_budget > 5: # Only attempt if some meaningful budget remains
            bench_players_in_squad = [p for p in selected_details if not p['is_starter']]
            bench_players_in_squad_sorted = sorted(bench_players_in_squad, key=lambda x: x['pvs']) # Lowest PVS bench first

            potential_upgrades_pool = eligible_df[~eligible_df['player_id'].isin([p['player_id'] for p in selected_details])].sort_values(by='pvs', ascending=False)

            for i, old_player_detail in enumerate(bench_players_in_squad_sorted):
                if current_budget <= 5 : break # Stop if budget too low for meaningful upgrades
                
                # Find better PVS players for the same position not in squad
                for _, new_player_row in potential_upgrades_pool[potential_upgrades_pool['simplified_position'] == old_player_detail['position']].iterrows():
                    if new_player_row['pvs'] > old_player_detail['pvs']:
                        cost_to_upgrade = new_player_row['mrb'] - old_player_detail['mrb_cost']
                        if cost_to_upgrade <= current_budget and cost_to_upgrade >= 0 : # Upgrade must be affordable and not free unless PVS gain
                            # Perform swap in selected_details
                            selected_details.pop(selected_details.index(old_player_detail)) # Remove old
                            selected_details.append({'player_id': new_player_row['player_id'], 'is_starter': False, 
                                                     'mrb_cost': new_player_row['mrb'], 'pvs': new_player_row['pvs'],
                                                     'position': new_player_row['simplified_position']})
                            current_budget -= cost_to_upgrade 
                            # (Positional counts remain the same as it's a same-position swap)
                            st.caption(f"Upgraded {df.loc[df['player_id'] == old_player_detail['player_id'], 'Joueur'].iloc[0]} to {new_player_row['Joueur']} (PVS gain, budget used: {cost_to_upgrade})")
                            break # Move to next bench player or re-evaluate (simple pass for MVP)
                    if current_budget <=5: break


        # --- Construct Final Squad DataFrame and Summary ---
        if not selected_details: return pd.DataFrame(), {}
        final_squad_ids = [p['player_id'] for p in selected_details]
        final_squad_df = df[df['player_id'].isin(final_squad_ids)].copy()
        details_df = pd.DataFrame(selected_details)
        final_squad_df = pd.merge(final_squad_df, details_df, on='player_id', how='left', suffixes=('', '_selection'))
        final_squad_df.rename(columns={'mrb_cost': 'mrb_actual_cost', 'pvs_selection': 'pvs_in_squad'}, inplace=True)
        if 'pvs_in_squad' not in final_squad_df.columns and 'pvs' in final_squad_df.columns : final_squad_df['pvs_in_squad'] = final_squad_df['pvs']
        
        # Ensure integer costs in final squad summary
        final_squad_df['mrb_actual_cost'] = final_squad_df['mrb_actual_cost'].round().astype(int)

        summary = {
            'total_players': len(final_squad_df),
            'total_cost': final_squad_df['mrb_actual_cost'].sum(),
            'remaining_budget': self.budget - final_squad_df['mrb_actual_cost'].sum(),
            'position_counts': final_squad_df['simplified_position'].value_counts().to_dict(),
            'total_squad_pvs': final_squad_df['pvs_in_squad'].sum(),
            'total_starters_pvs': final_squad_df[final_squad_df['is_starter']]['pvs_in_squad'].sum()
        }
        # Validation warnings
        return final_squad_df, summary

# --- Main Streamlit App UI ---
def main():
    st.markdown('<h1 class="main-header">🚀 MPG Auction Strategist v3</h1>', unsafe_allow_html=True)
    strategist = MPGAuctionStrategist()

    if 'current_profile_name' not in st.session_state:
        st.session_state.current_profile_name = "Balanced Value"
        profile_values = PREDEFINED_PROFILES[st.session_state.current_profile_name]
        st.session_state.n_recent = profile_values.get("n_recent_games", DEFAULT_N_RECENT_GAMES)
        st.session_state.min_recent_filter = profile_values.get("min_recent_games_played_filter", DEFAULT_MIN_RECENT_GAMES_PLAYED)
        st.session_state.kpi_weights = profile_values.get("kpi_weights", {})
        st.session_state.mrb_params_per_pos = profile_values.get("mrb_params_per_pos", {})
        st.session_state.formation_key = DEFAULT_FORMATION
        st.session_state.squad_size = DEFAULT_SQUAD_SIZE

    st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFr1EWtMR2tHq1FwHnCHqg2uXv1JMLYQlRZw&s", width=100)
    st.sidebar.markdown('<h2 class="section-header" style="margin-top:0;">⚙️ Controls & Settings</h2>', unsafe_allow_html=True)

    uploaded_file = st.sidebar.file_uploader("📁 Upload MPG Ratings File (CSV/Excel)", type=['csv', 'xlsx', 'xls'], help="Joueur, Poste, Club, Cote, %Titu, Indispo?, Gameweeks (D1..D34).")

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### Global Data & Form Parameters")
    n_recent_ui = st.sidebar.number_input("Recent Games Window (N)", min_value=1, max_value=38, value=st.session_state.n_recent, help="For 'Recent Form' KPIs. Avg of games *played* in this window.")
    min_recent_games_filter_ui = st.sidebar.number_input("Filter: Min Games Played in Recent N Weeks", min_value=0, max_value=n_recent_ui, value=st.session_state.min_recent_filter, help=f"Exclude players with < this many games in the '{n_recent_ui}' recent weeks. 0 = no filter.")
    if n_recent_ui != st.session_state.n_recent or min_recent_games_filter_ui != st.session_state.min_recent_filter:
        st.session_state.current_profile_name = "Custom"
    st.session_state.n_recent, st.session_state.min_recent_filter = n_recent_ui, min_recent_games_filter_ui

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 👥 Squad Building")
    formation_key_ui = st.sidebar.selectbox("Preferred Starting Formation", options=list(strategist.formations.keys()), index=list(strategist.formations.keys()).index(st.session_state.formation_key))
    target_squad_size_ui = st.sidebar.number_input("Target Total Squad Size", min_value=strategist.squad_minimums_sum_val, max_value=30, value=st.session_state.squad_size)
    if formation_key_ui != st.session_state.formation_key or target_squad_size_ui != st.session_state.squad_size:
        st.session_state.current_profile_name = "Custom"
    st.session_state.formation_key, st.session_state.squad_size = formation_key_ui, target_squad_size_ui

    st.sidebar.markdown("---")
    st.sidebar.markdown("#### 🎨 Settings Profiles")
    profile_names = list(PREDEFINED_PROFILES.keys())
    selected_profile_name_ui = st.sidebar.selectbox("Select Profile", options=profile_names, index=profile_names.index(st.session_state.current_profile_name), help="Loads predefined KPI weights & MRB params. Modifying details sets to 'Custom'.")
    if selected_profile_name_ui != st.session_state.current_profile_name:
        apply_profile(selected_profile_name_ui) # Function to update session_state from profile
        st.experimental_rerun() # Rerun to reflect changes in widgets below

    # Define apply_profile function (could be outside main or helper class)
    def apply_profile(profile_name_to_apply):
        if profile_name_to_apply != "Custom" and profile_name_to_apply in PREDEFINED_PROFILES:
            profile = PREDEFINED_PROFILES[profile_name_to_apply]
            st.session_state.n_recent = profile.get("n_recent_games", st.session_state.n_recent)
            st.session_state.min_recent_filter = profile.get("min_recent_games_played_filter", st.session_state.min_recent_filter)
            st.session_state.kpi_weights = profile.get("kpi_weights", st.session_state.kpi_weights)
            st.session_state.mrb_params_per_pos = profile.get("mrb_params_per_pos", st.session_state.mrb_params_per_pos)
        st.session_state.current_profile_name = profile_name_to_apply


    with st.sidebar.expander("📊 KPI Weights (0.0 to 1.0)", expanded=st.session_state.current_profile_name == "Custom"):
        active_kpi_weights = st.session_state.kpi_weights
        weights_ui = {}
        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            st.markdown(f'<h6>{pos}</h6>', unsafe_allow_html=True)
            default_pos_w = PREDEFINED_PROFILES["Balanced Value"]["kpi_weights"][pos]
            current_pos_w = active_kpi_weights.get(pos, default_pos_w)
            weights_ui[pos] = {
                'recent_avg': st.slider(f"Recent Avg Rating", 0.0,1.0,float(current_pos_w.get('recent_avg',0)),0.01,key=f"{pos}_wRA_dyn2"),
                'season_avg': st.slider(f"Season Avg Rating", 0.0,1.0,float(current_pos_w.get('season_avg',0)),0.01,key=f"{pos}_wSA_dyn2"),
                'regularity': st.slider(f"Regularity (%Titu)", 0.0,1.0,float(current_pos_w.get('regularity',0)),0.01,key=f"{pos}_wR_dyn2"),
                'recent_goals': st.slider(f"Recent Goals",0.0,1.0,float(current_pos_w.get('recent_goals',0)) if pos in ['MID','FWD'] else 0.0,0.01,key=f"{pos}_wRG_dyn2",disabled=pos not in ['MID','FWD']),
                'season_goals': st.slider(f"Season Goals",0.0,1.0,float(current_pos_w.get('season_goals',0)) if pos in ['MID','FWD'] else 0.0,0.01,key=f"{pos}_wSG_dyn2",disabled=pos not in ['MID','FWD'])
            }
        if weights_ui != active_kpi_weights: st.session_state.current_profile_name = "Custom"
        st.session_state.kpi_weights = weights_ui

    with st.sidebar.expander("💰 MRB Parameters (Per Position)", expanded=st.session_state.current_profile_name == "Custom"):
        active_mrb_params = st.session_state.mrb_params_per_pos
        mrb_params_ui = {}
        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            st.markdown(f'<h6>{pos}</h6>', unsafe_allow_html=True)
            default_pos_mrb = PREDEFINED_PROFILES["Balanced Value"]["mrb_params_per_pos"][pos]
            current_pos_mrb = active_mrb_params.get(pos, default_pos_mrb)
            mrb_params_ui[pos] = {
                'max_proportional_bonus_at_pvs100': st.slider(f"Max Bonus Factor (at PVS 100)", 0.0, 1.0, float(current_pos_mrb.get('max_proportional_bonus_at_pvs100',0.2)), 0.01, key=f"{pos}_mrbMPB_dyn2", help="Bonus if PVS=100 (0.5 = 50% bonus, MRB up to 1.5x Cote). MRB capped at 2x Cote overall.")
            }
        if mrb_params_ui != active_mrb_params: st.session_state.current_profile_name = "Custom"
        st.session_state.mrb_params_per_pos = mrb_params_ui

    # --- Dynamic Calculation ---
    if uploaded_file:
        # All inputs should now be in st.session_state
        with st.spinner("🧠 Recalculating squad based on settings..."):
            try:
                df_input_calc = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded_file)
                df_processed_calc = df_input_calc.copy()
                df_processed_calc['simplified_position'] = df_processed_calc['Poste'].apply(strategist.simplify_position)
                df_processed_calc['player_id'] = df_processed_calc.apply(strategist.create_player_id, axis=1)
                df_processed_calc['Cote'] = pd.to_numeric(df_processed_calc['Cote'], errors='coerce').fillna(1).clip(lower=1).round().astype(int)
                if 'Indispo ?' not in df_processed_calc.columns: df_processed_calc['Indispo ?'] = False
                else: df_processed_calc['Indispo ?'] = df_processed_calc['Indispo ?'].astype(str).str.upper().isin(['TRUE', 'OUI', '1', 'YES', 'VRAI'])

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
                # st.exception(e) # For dev debugging

    # --- Main Panel Display Logic (uses updated session state keys) ---
    # (Display logic for squad, summary, and full player list as in previous response, ensuring it uses the '_final' suffixed session state keys)
    if 'squad_df_result_final' in st.session_state and \
       st.session_state['squad_df_result_final'] is not None and \
       not st.session_state['squad_df_result_final'].empty:
        
        col_main_results, col_summary_sidebar = st.columns([3, 1]) # Main results, then summary
        with col_main_results:
            st.markdown('<h2 class="section-header">🏆 Suggested Squad</h2>', unsafe_allow_html=True)
            # (DataFrame display logic from previous response - no starter styling)
            squad_display_df = st.session_state['squad_df_result_final'].copy()
            # Ensure MRB, Cote, goals are int for display, others rounded
            squad_display_df['Cote'] = pd.to_numeric(squad_display_df['Cote'], errors='coerce').fillna(0).round().astype(int)
            squad_display_df['mrb_actual_cost'] = pd.to_numeric(squad_display_df['mrb_actual_cost'], errors='coerce').fillna(0).round().astype(int)
            for col in ['recent_goals', 'season_goals']:
                if col in squad_display_df.columns:
                    squad_display_df[col] = pd.to_numeric(squad_display_df[col], errors='coerce').fillna(0).round().astype(int)

            squad_display_cols = ['Joueur', 'Club', 'simplified_position', 'is_starter', 
                                  'mrb_actual_cost', 'Cote', 'pvs_in_squad', 
                                  'recent_avg_rating', 'season_avg_rating', '%Titu',
                                  'recent_goals', 'season_goals', 'value_per_cost'] # value_per_cost uses MRB
            display_squad_cols_exist = [col for col in squad_display_cols if col in squad_display_df.columns]
            squad_display_df = squad_display_df[display_squad_cols_exist]
            
            squad_display_df.rename(columns={
                'Joueur': 'Player', 'simplified_position': 'Pos', 'is_starter': 'Starter',
                'mrb_actual_cost': 'MRB (Cost Paid)', 'Cote': 'Listed Price', 'pvs_in_squad': 'PVS (0-100)',
                'recent_avg_rating': 'Rec.Avg.R (0-10)', 'season_avg_rating': 'Sea.Avg.R (0-10)',
                '%Titu': 'Regularity %', 'recent_goals': 'Rec.Goals (N)', 'season_goals': 'Sea.Goals',
                'value_per_cost': 'PVS/MRB Ratio'
            }, inplace=True)
            
            for col_name_disp in ['PVS (0-100)', 'Rec.Avg.R (0-10)', 'Sea.Avg.R (0-10)', 'PVS/MRB Ratio', 'Regularity %']:
                if col_name_disp in squad_display_df.columns: 
                    squad_display_df[col_name_disp] = pd.to_numeric(squad_display_df[col_name_disp], errors='coerce').fillna(0.0).round(2)
            
            pos_order = ['GK', 'DEF', 'MID', 'FWD']
            if 'Pos' in squad_display_df.columns:
                squad_display_df['Pos'] = pd.Categorical(squad_display_df['Pos'], categories=pos_order, ordered=True)
                squad_display_df = squad_display_df.sort_values(by=['Starter', 'Pos', 'PVS (0-100)'], ascending=[False, True, False])
            
            st.dataframe(squad_display_df, use_container_width=True, hide_index=True) # No starter styling


        with col_summary_sidebar:
            st.markdown('<h2 class="section-header">📈 Squad Summary</h2>', unsafe_allow_html=True)
            summary = st.session_state['squad_summary_result_final']
            if summary and isinstance(summary, dict):
                st.metric("Budget Spent (MRB)", f"€ {summary.get('total_cost',0):.0f} / {strategist.budget}", help=f"Remaining: € {summary.get('remaining_budget',0):.0f}")
                st.metric("Final Squad Size", f"{summary.get('total_players',0)} players (Target: {st.session_state.squad_size})")
                st.metric("Total Squad PVS", f"{summary.get('total_squad_pvs',0):.2f}")
                st.metric("Starters PVS", f"{summary.get('total_starters_pvs',0):.2f}")
                st.info(f"**Built for Formation:** {st.session_state.get('selected_formation_key_display_final', 'N/A')}")
                st.markdown("**Actual Positional Breakdown:**")
                for pos_cat in pos_order:
                    count = summary.get('position_counts', {}).get(pos_cat, 0)
                    min_req = strategist.squad_minimums.get(pos_cat,0)
                    st.write(f"• **{pos_cat}:** {count} (Min: {min_req})")
            else: st.warning("Squad summary not available.")
        
        st.markdown('<hr><h2 class="section-header">📋 Full Player Database & Calculated Values</h2>', unsafe_allow_html=True)
        if 'df_for_display_final' in st.session_state and st.session_state['df_for_display_final'] is not None:
            # Display logic for full player list (as in previous response, ensuring integer display for Cote, MRB, Goals)
            df_full_display = st.session_state['df_for_display_final'].copy()
            
            int_cols_full_display = ['Cote', 'mrb', 'recent_goals', 'season_goals', 'recent_games_played_count']
            for col in int_cols_full_display:
                if col in df_full_display.columns:
                     df_full_display[col] = pd.to_numeric(df_full_display[col], errors='coerce').fillna(0).round().astype(int)

            all_stats_cols = [
                'Joueur', 'Club', 'simplified_position', 'Poste','Indispo ?', 'Cote', 'pvs', 'mrb', 'value_per_cost',
                'recent_avg_rating', 'season_avg_rating', '%Titu', 'recent_goals', 'season_goals', 'recent_games_played_count',
                'norm_recent_avg', 'norm_season_avg', 'norm_regularity', 'norm_recent_goals', 'norm_season_goals'
            ]
            display_all_stats_cols_exist = [col for col in all_stats_cols if col in df_full_display.columns]
            df_full_display = df_full_display[display_all_stats_cols_exist]

            df_full_display.rename(columns={
                'Joueur': 'Player', 'simplified_position': 'Pos','Poste':'Orig.Pos', 'Indispo ?': 'Unavail.',
                'Cote': 'Price', 'pvs': 'PVS', 'mrb': 'MRB', 'value_per_cost': 'Val/MRB',
                'recent_avg_rating': 'Rec.AvgR', 'season_avg_rating': 'Sea.AvgR',
                '%Titu': 'Reg.%', 'recent_goals': 'Rec.G', 'season_goals': 'Sea.G',
                'recent_games_played_count': 'Rec.Plyd',
                'norm_recent_avg': 'N.RecAvg', 'norm_season_avg': 'N.SeaAvg',
                'norm_regularity': 'N.Reg%', 'norm_recent_goals': 'N.RecG', 'norm_season_goals': 'N.SeaG'
            }, inplace=True)
            
            for col_name_disp_full in df_full_display.columns: # Round floats
                if 'Price' in col_name_disp_full or 'MRB' in col_name_disp_full or 'G' == col_name_disp_full[-1] or 'Plyd' in col_name_disp_full or 'Reg.%' == col_name_disp_full :
                    pass # Already int or handled by %Titu
                elif 'PVS' in col_name_disp_full or 'Ratio' in col_name_disp_full or 'Avg' in col_name_disp_full :
                    if col_name_disp_full in df_full_display.columns: df_full_display[col_name_disp_full] = pd.to_numeric(df_full_display[col_name_disp_full], errors='coerce').fillna(0.0).round(2)


            search_term_all = st.text_input("🔍 Search All Players:", key="search_all_players_input_key_dyn2")
            if search_term_all:
                df_full_display = df_full_display[df_full_display.apply(lambda row: row.astype(str).str.contains(search_term_all, case=False, na=False).any(), axis=1)]
            
            st.dataframe(df_full_display.sort_values(by='PVS', ascending=False), use_container_width=True, hide_index=True, height=600)
            st.download_button(label="📥 Download Full Player Analysis (CSV)", data=df_full_display.to_csv(index=False).encode('utf-8'), file_name="mpg_full_player_analysis_v3.csv", mime="text/csv", key="download_v3")
        
        elif not uploaded_file: pass 
        elif 'squad_df_result_final' not in st.session_state and uploaded_file :
             st.info("📊 Adjust settings. Results update dynamically.")
    else:
        st.info("👈 Upload your MPG ratings file to begin.")
        # Display Expected File Format Guide (as before)

if __name__ == "__main__":
    main()
