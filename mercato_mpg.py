import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional

# Page configuration
st.set_page_config(
    page_title="MPG Auction Strategist",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4; /* Primary color */
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Arial', sans-serif; /* Example font */
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e7d32; /* Secondary color */
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2e7d32;
        padding-bottom: 0.3rem;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 0.5rem;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #155a8a;
        color: white;
    }
    .stSlider [data-baseweb="slider"] {
        padding-bottom: 10px; /* More space for slider labels */
    }
    /* Ensure sidebar content is not overly cramped */
    .css-1d391kg {
        padding-top: 2rem; /* Adjust sidebar top padding */
    }
</style>
""", unsafe_allow_html=True)

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
            # st.warning(f"Unknown position '{position}' encountered. Please map it.") # Optional: Log unknown positions during processing
            return 'UNKNOWN'

    def create_player_id(self, row) -> str:
        name = str(row.get('Joueur', '')).strip()
        simplified_pos = self.simplify_position(row.get('Poste', ''))
        club = str(row.get('Club', '')).strip()
        return f"{name}_{simplified_pos}_{club}"

    def extract_rating_and_goals(self, rating_str) -> Tuple[Optional[float], int, bool]:
        """Extract MPG rating, goals, and if played (not DNP: not blank, not '0' as string)"""
        # User clarified: "0 or blank is did not play"
        if pd.isna(rating_str) or str(rating_str).strip() == '' or str(rating_str).strip() == '0':
            return None, 0, False # DNP
        
        rating_val_str = str(rating_str).strip()
        goals = rating_val_str.count('*')
        clean_rating_str = re.sub(r'[()\*]', '', rating_val_str)
        
        try:
            rating = float(clean_rating_str)
            # User treats "0" rating as DNP. If after cleaning it's still 0, it means it was '0' without '()' or '*'
            # This was already handled by the initial check. So, any valid float here means they played.
            return rating, goals, True # Played
        except ValueError:
            return None, 0, False # If conversion fails, treat as DNP

    def get_gameweek_columns(self, df_columns: List[str]) -> List[str]:
        """Identify and sort gameweek columns (e.g., D1, D2, ..., D34 with D34 as most recent)"""
        gw_cols_data = []
        for col in df_columns:
            match = re.fullmatch(r'D(\d+)', col) # Matches D<number> only (e.g., D1, D34), ignores D-34
            if match:
                gw_cols_data.append({'name': col, 'number': int(match.group(1))})
        
        # Sort by gameweek number (ascending, so D1 is first, D34 is last)
        # This makes it easier to take the N most recent by slicing from the end.
        sorted_gw_cols_data = sorted(gw_cols_data, key=lambda x: x['number'])
        return [col['name'] for col in sorted_gw_cols_data]

    def calculate_kpis(self, df: pd.DataFrame, n_recent: int) -> pd.DataFrame:
        result_df = df.copy()
        
        all_df_gameweek_cols_sorted = self.get_gameweek_columns(df.columns) # Sorted D1, D2, ..., D34
        
        kpi_cols_to_init = [
            'recent_avg_rating', 'recent_goals', 'season_avg_rating', 'season_goals',
            'recent_games_played_count'
        ]
        for col in kpi_cols_to_init:
            result_df[col] = 0.0
            if 'count' in col: result_df[col] = 0

        for idx, row in result_df.iterrows():
            season_ratings_when_played = []
            season_goals_total = 0
            
            for gw_col_name in all_df_gameweek_cols_sorted:
                rating, goals, played_this_gw = self.extract_rating_and_goals(row.get(gw_col_name)) # Use .get for safety
                if played_this_gw and rating is not None:
                    season_ratings_when_played.append(rating)
                    season_goals_total += goals
            
            result_df.at[idx, 'season_avg_rating'] = np.mean(season_ratings_when_played) if season_ratings_when_played else 0
            result_df.at[idx, 'season_goals'] = season_goals_total

            # Recent form: based on N most recent *calendar* gameweeks.
            # Average is of games *played* within that window.
            # Goals are sum of goals *within* that window (0 if not played).
            recent_calendar_gws_to_check = all_df_gameweek_cols_sorted[-n_recent:] if len(all_df_gameweek_cols_sorted) >= n_recent else all_df_gameweek_cols_sorted
            
            recent_ratings_when_played_in_window = []
            recent_goals_in_window = 0
            recent_games_played_in_window_count = 0

            for gw_col_name in recent_calendar_gws_to_check:
                rating, goals, played_this_gw = self.extract_rating_and_goals(row.get(gw_col_name))
                if played_this_gw and rating is not None:
                    recent_ratings_when_played_in_window.append(rating)
                    recent_goals_in_window += goals # Only add goals if played
                    recent_games_played_in_window_count += 1
            
            result_df.at[idx, 'recent_avg_rating'] = np.mean(recent_ratings_when_played_in_window) if recent_ratings_when_played_in_window else 0
            result_df.at[idx, 'recent_goals'] = recent_goals_in_window
            result_df.at[idx, 'recent_games_played_count'] = recent_games_played_in_window_count
        
        return result_df

    def normalize_kpis(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        
        result_df['norm_recent_avg'] = np.clip(result_df['recent_avg_rating'] * 10, 0, 100)
        result_df['norm_season_avg'] = np.clip(result_df['season_avg_rating'] * 10, 0, 100)
        result_df['norm_regularity'] = pd.to_numeric(result_df['%Titu'], errors='coerce').fillna(0).clip(0, 100)
        
        result_df['norm_recent_goals'] = 0.0
        result_df['norm_season_goals'] = 0.0

        for pos in ['MID', 'FWD']:
            pos_mask = result_df['simplified_position'] == pos
            if pos_mask.sum() > 0:
                # Recent goals: 5+ in N played games = 100 (as per user rule)
                # The KPI 'recent_goals' is sum of goals in played games in recent N calendar weeks.
                result_df.loc[pos_mask, 'norm_recent_goals'] = np.clip(result_df.loc[pos_mask, 'recent_goals'] * 20, 0, 100)
                
                max_season_goals = result_df.loc[pos_mask, 'season_goals'].max()
                if max_season_goals > 0:
                    result_df.loc[pos_mask, 'norm_season_goals'] = np.clip(
                        (result_df.loc[pos_mask, 'season_goals'] / max_season_goals * 100), 0, 100
                    )
                else:
                    result_df.loc[pos_mask, 'norm_season_goals'] = 0
        return result_df

    def calculate_pvs(self, df: pd.DataFrame, weights: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        result_df = df.copy()
        result_df['pvs'] = 0.0 # Initialize PVS column
        
        for pos_simplified in ['GK', 'DEF', 'MID', 'FWD']:
            pos_mask = result_df['simplified_position'] == pos_simplified
            if not pos_mask.any():
                continue
            
            pos_weights = weights[pos_simplified]
            current_pvs_sum = pd.Series(0.0, index=result_df.loc[pos_mask].index)

            current_pvs_sum += result_df.loc[pos_mask, 'norm_recent_avg'].fillna(0) * pos_weights.get('recent_avg', 0)
            current_pvs_sum += result_df.loc[pos_mask, 'norm_season_avg'].fillna(0) * pos_weights.get('season_avg', 0)
            current_pvs_sum += result_df.loc[pos_mask, 'norm_regularity'].fillna(0) * pos_weights.get('regularity', 0)
            
            if pos_simplified in ['MID', 'FWD']:
                current_pvs_sum += result_df.loc[pos_mask, 'norm_recent_goals'].fillna(0) * pos_weights.get('recent_goals', 0)
                current_pvs_sum += result_df.loc[pos_mask, 'norm_season_goals'].fillna(0) * pos_weights.get('season_goals', 0)
            
            # PVS is now sum of (KPI_0_100 * weight_0_1), so scale is effectively 0-100 if weights sum to 1.
            # User wants PVS on 0-100 scale.
            result_df.loc[pos_mask, 'pvs'] = current_pvs_sum.clip(0,100) # Clip to ensure it's within 0-100
                                                                    # (could exceed if sum of weights > 1)

        return result_df

    def calculate_mrb(self, df: pd.DataFrame, mrb_params: Dict) -> pd.DataFrame:
        result_df = df.copy()
        
        baseline_pvs_0_100 = mrb_params['baseline_pvs_0_100']
        max_markup_pct_val = mrb_params['max_markup_pct'] / 100.0
        points_for_max_markup_pvs = mrb_params['points_for_max_markup_0_100']
        absolute_max_bid_val = mrb_params['absolute_max_bid']
        
        def calc_mrb_dynamic(row):
            cote = row['Cote']
            pvs = row['pvs'] # PVS is 0-100
            
            if pd.isna(cote) or cote <= 0: cote = 1.0 # Ensure cote is a positive float
            
            actual_markup_percentage = 0.0
            if pvs > baseline_pvs_0_100:
                excess_pvs = pvs - baseline_pvs_0_100
                if points_for_max_markup_pvs > 0: # Avoid division by zero
                    actual_markup_percentage = min(max_markup_pct_val, (excess_pvs / points_for_max_markup_pvs) * max_markup_pct_val)
                elif excess_pvs > 0 : # if points_for_max_markup is 0, any excess PVS gets max markup
                    actual_markup_percentage = max_markup_pct_val

            mrb = cote * (1 + actual_markup_percentage)
            mrb = min(mrb, absolute_max_bid_val)
            return round(float(mrb))

        result_df['mrb'] = result_df.apply(calc_mrb_dynamic, axis=1)
        
        safe_mrb = result_df['mrb'].replace(0, np.nan).astype(float) # Ensure float for division
        result_df['value_per_cost'] = result_df['pvs'] / safe_mrb
        result_df['value_per_cost'].fillna(0, inplace=True)
        
        return result_df

    def select_squad(self, df: pd.DataFrame, formation_key: str, target_squad_size: int,
                     min_recent_games_played: int) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        
        # 1. Apply minimum recent games played filter (from N recent calendar weeks)
        if min_recent_games_played > 0:
            eligible_df = df[df['recent_games_played_count'] >= min_recent_games_played].copy()
        else:
            eligible_df = df.copy()

        # 2. Filter out unavailable players
        # Ensure 'Indispo ?' column exists. If not, assume all are available.
        if 'Indispo ?' in eligible_df.columns:
             eligible_df = eligible_df[~eligible_df['Indispo ?'].astype(str).str.upper().isin(['TRUE', 'OUI', '1', 'YES'])].copy()
        
        if eligible_df.empty:
            st.warning("No eligible players available after filtering (availability/recent games). Adjust filters or check data.")
            return None, None

        selected_player_ids = []
        current_budget_spent = 0
        # Tracks actual players added to squad for each simplified position
        squad_pos_counts = {pos: 0 for pos in ['GK', 'DEF', 'MID', 'FWD']}
        
        # Ensure player_id is unique for selection pool
        eligible_df = eligible_df.drop_duplicates(subset=['player_id'])
        
        # Store dicts of selected players with their role for easier processing later
        squad_selection_details = []


        # --- Phase 1: Select Starters for Preferred Formation ---
        starters_needed_map = self.formations[formation_key].copy()
        
        for pos, num_to_select in starters_needed_map.items():
            candidates = eligible_df[
                (eligible_df['simplified_position'] == pos) &
                (~eligible_df['player_id'].isin(selected_player_ids))
            ].sort_values(by='pvs', ascending=False) # Prioritize PVS for starters
            
            selected_for_this_pos_slot = 0
            for _, player_row in candidates.iterrows():
                if selected_for_this_pos_slot >= num_to_select:
                    break
                if current_budget_spent + player_row['mrb'] <= self.budget:
                    selected_player_ids.append(player_row['player_id'])
                    squad_selection_details.append({'player_id': player_row['player_id'], 'is_starter': True, 'mrb_cost': player_row['mrb'], 'position': pos})
                    current_budget_spent += player_row['mrb']
                    squad_pos_counts[pos] += 1
                    selected_for_this_pos_slot += 1
        
        # --- Phase 2: Fulfill Overall Squad Positional Minimums ---
        for pos, overall_min in self.squad_minimums.items():
            needed_for_overall_min = max(0, overall_min - squad_pos_counts[pos])
            if needed_for_overall_min == 0:
                continue

            candidates = eligible_df[
                (eligible_df['simplified_position'] == pos) &
                (~eligible_df['player_id'].isin(selected_player_ids))
            ].sort_values(by='value_per_cost', ascending=False)
            
            selected_for_this_pos_slot = 0
            for _, player_row in candidates.iterrows():
                if selected_for_this_pos_slot >= needed_for_overall_min:
                    break
                if current_budget_spent + player_row['mrb'] <= self.budget:
                    selected_player_ids.append(player_row['player_id'])
                    squad_selection_details.append({'player_id': player_row['player_id'], 'is_starter': False, 'mrb_cost': player_row['mrb'], 'position': pos})
                    current_budget_spent += player_row['mrb']
                    squad_pos_counts[pos] += 1
                    selected_for_this_pos_slot += 1

        # --- Phase 3: Complete the Squad to Total Squad Size ---
        current_total_players_in_squad = len(selected_player_ids)
        remaining_slots_to_fill_total = max(0, target_squad_size - current_total_players_in_squad)

        if remaining_slots_to_fill_total > 0:
            candidates = eligible_df[
                (~eligible_df['player_id'].isin(selected_player_ids))
            ].sort_values(by='value_per_cost', ascending=False)
            
            selected_for_this_phase = 0
            for _, player_row in candidates.iterrows():
                if selected_for_this_phase >= remaining_slots_to_fill_total:
                    break
                # Ensure we don't add a player if it violates a max (though we don't have max per pos rule for now)
                # or if it makes no sense (e.g. trying to add 3rd GK when only 2 are useful)
                # For now, just fill up to squad size.
                if current_budget_spent + player_row['mrb'] <= self.budget:
                    selected_player_ids.append(player_row['player_id'])
                    squad_selection_details.append({'player_id': player_row['player_id'], 'is_starter': False, 'mrb_cost': player_row['mrb'], 'position': player_row['simplified_position']})
                    current_budget_spent += player_row['mrb']
                    squad_pos_counts[player_row['simplified_position']] += 1
                    selected_for_this_phase += 1
        
        # --- Construct Final Squad DataFrame and Summary ---
        if not selected_player_ids:
            st.warning("Could not select any players based on current criteria and budget.")
            return None, None

        final_squad_df_from_selection = df[df['player_id'].isin(selected_player_ids)].copy()
        
        # Add 'is_starter' status and 'mrb_cost_in_squad' to the DataFrame for display
        starter_map = {item['player_id']: item['is_starter'] for item in squad_selection_details}
        mrb_cost_map = {item['player_id']: item['mrb_cost'] for item in squad_selection_details}

        final_squad_df_from_selection['is_starter'] = final_squad_df_from_selection['player_id'].map(starter_map)
        final_squad_df_from_selection['mrb_actual_cost'] = final_squad_df_from_selection['player_id'].map(mrb_cost_map)


        squad_summary = {
            'total_players': len(final_squad_df_from_selection),
            'total_cost': final_squad_df_from_selection['mrb_actual_cost'].sum(), # Use MRBs of selected players
            'remaining_budget': self.budget - final_squad_df_from_selection['mrb_actual_cost'].sum(),
            'position_counts': final_squad_df_from_selection['simplified_position'].value_counts().to_dict()
        }

        # Final validation checks for squad rules
        for pos, min_val in self.squad_minimums.items():
            if squad_summary['position_counts'].get(pos, 0) < min_val:
                st.warning(f"Warning: Squad minimum for {pos} not met ({squad_summary['position_counts'].get(pos, 0)} selected, {min_val} required). Budget may have been too restrictive or player pool too small after filtering.")
        if squad_summary['total_players'] < self.squad_minimums_sum_val :
             st.warning(f"Warning: Total selected players ({squad_summary['total_players']}) is less than the sum of positional minimums ({self.squad_minimums_sum_val}).")
        elif squad_summary['total_players'] < target_squad_size :
             st.warning(f"Warning: Could only select {squad_summary['total_players']} players out of target {target_squad_size} due to budget or player availability.")


        return final_squad_df_from_selection, squad_summary

# --- Main Streamlit App UI ---
def main():
    st.markdown('<h1 class="main-header">⚽ MPG Auction Strategist</h1>', unsafe_allow_html=True)
    
    strategist = MPGAuctionStrategist() #
    
    # --- Sidebar ---
    st.sidebar.image("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRFr1EWtMR2tHq1FwHnCHqg2uXv1JMLYQlRZw&s", width=100) # Example icon/logo
    st.sidebar.markdown('<h2 class="section-header" style="margin-top:0;">⚙️ Settings & Controls</h2>', unsafe_allow_html=True)

    with st.sidebar.expander("📁 File Upload & Data Settings", expanded=True):
        uploaded_file = st.file_uploader( #
            "Upload MPG Ratings File (CSV/Excel)", #
            type=['csv', 'xlsx', 'xls'],
            help="Ensure columns: Joueur, Poste, Club, Cote, %Titu, Indispo ? (optional), and Gameweeks (e.g., D1...D34)."
        )
        n_recent = st.number_input( #
            "Recent Games Window (N)", #
            min_value=1, max_value=38, value=5,
            help="Number of most recent calendar gameweeks for 'Recent Form' KPIs. Avg rating is from games *played* within this window." #
        )
        min_recent_games_played_filter = st.number_input( #
            "Filter: Min Games Played in Last N Weeks", #
            min_value=0, max_value=n_recent, value=0, # Default 0 = no filter
            help=f"Exclude players who played in fewer than this many games within the '{n_recent}' recent calendar weeks window. '0' disables this filter." #
        )

    if uploaded_file is not None:
        try:
            df_input = pd.read_excel(uploaded_file) if uploaded_file.name.endswith(('.xlsx', '.xls')) else pd.read_csv(uploaded_file) #
            
            # --- Basic Data Validation ---
            required_cols = ['Joueur', 'Poste', 'Club', 'Cote', '%Titu']
            missing_cols = [col for col in required_cols if col not in df_input.columns]
            if missing_cols:
                st.error(f"Error: Missing required columns in uploaded file: {', '.join(missing_cols)}. Please check the file format guide below.")
                return # Stop further processing if essential columns are missing
            
            # --- Initial Data Processing ---
            df_processed = df_input.copy()
            df_processed['simplified_position'] = df_processed['Poste'].apply(strategist.simplify_position)
            unknown_pos_count = (df_processed['simplified_position'] == 'UNKNOWN').sum()
            if unknown_pos_count > 0:
                st.warning(f"{unknown_pos_count} players have an 'UNKNOWN' position after simplification. Review 'Poste' column mapping.")
            
            df_processed['player_id'] = df_processed.apply(strategist.create_player_id, axis=1)
            df_processed['Cote'] = pd.to_numeric(df_processed['Cote'], errors='coerce').fillna(1).clip(lower=1) # Ensure Cote is at least 1
            
            # Handle 'Indispo ?' column carefully
            if 'Indispo ?' not in df_processed.columns:
                st.sidebar.warning("Column 'Indispo ?' not found. Assuming all players are available.")
                df_processed['Indispo ?'] = False # Assume available
            else:
                # Convert various "unavailable" markers to a boolean True
                unavailable_markers = ['TRUE', 'OUI', '1', 'YES', 'VRAI'] # Add more if needed
                df_processed['Indispo ?'] = df_processed['Indispo ?'].astype(str).str.upper().isin(unavailable_markers)

            st.sidebar.success(f"✅ File '{uploaded_file.name}' loaded ({len(df_processed)} players).")

            # --- UI Sections Reordered as per User Request ---
            with st.sidebar.expander("👥 Squad Building Parameters", expanded=True):
                formation_key = st.selectbox(
                    "Preferred Starting Formation",
                    options=list(strategist.formations.keys()),
                    index=list(strategist.formations.keys()).index("4-4-2"), # Default to 4-4-2
                    help="Select the primary starting formation the algorithm will try to build strong starters for."
                )
                target_squad_size = st.number_input(
                    "Target Total Squad Size",
                    min_value=strategist.squad_minimums_sum_val, max_value=30, value=20,
                    help=f"Number of players for the full squad (Min {strategist.squad_minimums_sum_val} based on 2GK,6D,6M,4F; Max 30)."
                )

            with st.sidebar.expander("📊 KPI Weights (0.0 to 1.0)", expanded=True):
                st.markdown("Adjust the relative importance of each factor for Player Value Score (PVS).")
                weights = {}
                default_weights = {
                    'GK':  {'recent_avg': 0.4, 'season_avg': 0.4, 'regularity': 0.2, 'recent_goals': 0.0, 'season_goals': 0.0},
                    'DEF': {'recent_avg': 0.3, 'season_avg': 0.3, 'regularity': 0.4, 'recent_goals': 0.0, 'season_goals': 0.0},
                    'MID': {'recent_avg': 0.25, 'season_avg': 0.25, 'regularity': 0.2, 'recent_goals': 0.15, 'season_goals': 0.15},
                    'FWD': {'recent_avg': 0.25, 'season_avg': 0.25, 'regularity': 0.1, 'recent_goals': 0.2, 'season_goals': 0.2}
                }
                for pos_simplified in ['GK', 'DEF', 'MID', 'FWD']:
                    st.markdown(f'<h4>{pos_simplified}</h4>', unsafe_allow_html=True)
                    dw = default_weights[pos_simplified]
                    weights[pos_simplified] = {
                        'recent_avg': st.slider(f"Recent Avg Rating", 0.0, 1.0, dw['recent_avg'], 0.01, key=f"{pos_simplified}_w_rec_avg", help="Importance of avg rating in recent N played games."),
                        'season_avg': st.slider(f"Season Avg Rating", 0.0, 1.0, dw['season_avg'], 0.01, key=f"{pos_simplified}_w_sea_avg", help="Importance of avg rating over all played games this season."),
                        'regularity': st.slider(f"Regularity (%Titu)", 0.0, 1.0, dw['regularity'], 0.01, key=f"{pos_simplified}_w_reg", help="Importance of %Titu (titularisation rate)."),
                        'recent_goals': st.slider(f"Recent Goals", 0.0, 1.0, dw['recent_goals'], 0.01, key=f"{pos_simplified}_w_rec_g", help="Importance of goals in recent N played games (for MID/FWD)."),
                        'season_goals': st.slider(f"Season Goals", 0.0, 1.0, dw['season_goals'], 0.01, key=f"{pos_simplified}_w_sea_g", help="Importance of total season goals (for MID/FWD).")
                    }
            
            with st.sidebar.expander("💰 MRB Calculation Parameters", expanded=True):
                st.markdown("Configure how 'Max Recommended Bid' (MRB) is derived from PVS (0-100 scale) and Cote.")
                mrb_params = { # Adjusted for PVS 0-100 scale
                    'baseline_pvs_0_100': st.number_input("PVS Baseline (0-100)", min_value=0, max_value=100, value=55, step=1, help="PVS score a player needs to be 'worth' their Cote. MRB defaults to Cote if PVS is below this."),
                    'max_markup_pct': st.number_input("Max Markup % over Cote", min_value=0, max_value=200, value=30, step=5, help="Max % willing to bid over Cote for high PVS players."),
                    'points_for_max_markup_0_100': st.number_input("PVS points (above baseline) for Max Markup", min_value=1, max_value=50, value=25, step=1, help="How many PVS points above baseline are needed to apply full 'Max Markup %'."),
                    'absolute_max_bid': st.number_input("Absolute Max Bid (any player)", min_value=1, max_value=300, value=150, step=5, help="The absolute highest MRB the app will suggest for any single player, regardless of Cote or PVS.")
                }

            if st.sidebar.button("🚀 Calculate Optimal Squad & MRBs", type="primary", use_container_width=True):
                with st.spinner("🧠 Strategizing your optimal squad... Hang tight!"):
                    try:
                        df_kpis = strategist.calculate_kpis(df_processed, n_recent)
                        df_norm_kpis = strategist.normalize_kpis(df_kpis)
                        df_pvs = strategist.calculate_pvs(df_norm_kpis, weights)
                        df_mrb = strategist.calculate_mrb(df_pvs, mrb_params)
                        
                        squad_df_result, squad_summary_result = strategist.select_squad(df_mrb, formation_key, target_squad_size, min_recent_games_played_filter)
                        
                        st.session_state['df_for_display'] = df_mrb # Save for full list
                        st.session_state['squad_df_result'] = squad_df_result
                        st.session_state['squad_summary_result'] = squad_summary_result
                        st.session_state['selected_formation_key'] = formation_key
                        st.success("✅ Squad calculation complete! Results are displayed.")
                    except Exception as e:
                        st.error(f"💥 Error during calculation: {str(e)}")
                        st.exception(e) # Shows full traceback

            # --- Main Panel for Results ---
            if 'squad_df_result' in st.session_state and st.session_state['squad_df_result'] is not None and not st.session_state['squad_df_result'].empty:
                col_main_results, col_summary_sidebar = st.columns([3, 1]) # Main results, then summary

                with col_main_results:
                    st.markdown('<h2 class="section-header">🏆 Suggested Squad</h2>', unsafe_allow_html=True)
                    
                    squad_display_df = st.session_state['squad_df_result'].copy()
                    squad_display_cols = [
                        'Joueur', 'Club', 'simplified_position', 'is_starter', 
                        'mrb_actual_cost', 'Cote', 'pvs', 
                        'recent_avg_rating', 'season_avg_rating', '%Titu',
                        'recent_goals', 'season_goals',
                        'value_per_cost'
                    ]
                    display_squad_cols_exist = [col for col in squad_display_cols if col in squad_display_df.columns]
                    squad_display_df = squad_display_df[display_squad_cols_exist]

                    squad_display_df.rename(columns={
                        'Joueur': 'Player', 'simplified_position': 'Pos', 'is_starter': 'Starter',
                        'mrb_actual_cost': 'MRB (Cost Paid)', 'Cote': 'Listed Price', 'pvs': 'PVS (0-100)',
                        'recent_avg_rating': 'Rec.Avg.Rate (0-10)', 'season_avg_rating': 'Sea.Avg.Rate (0-10)',
                        '%Titu': 'Regularity %', 'recent_goals': 'Rec.Goals (N games)', 'season_goals': 'Sea.Goals',
                        'value_per_cost': 'PVS/MRB Ratio'
                    }, inplace=True)
                    
                    Apply styling for starter rows
                    def highlight_starters(row):
                       return ['background-color: #e8f5e8'] * len(row) if row.Starter else [''] * len(row)
                    
                    st.dataframe(
                        squad_display_df.style.apply(highlight_starters, axis=1),
                        use_container_width=True, 
                        hide_index=True
                    )

                with col_summary_sidebar:
                    st.markdown('<h2 class="section-header">📈 Squad Summary</h2>', unsafe_allow_html=True)
                    summary = st.session_state['squad_summary_result']
                    if summary:
                        st.metric("Budget Spent (MRB)", f"€ {summary['total_cost']:.0f} / {strategist.budget}", 
                                  help=f"Remaining Budget: € {summary['remaining_budget']:.0f}", delta_color="inverse")
                        st.metric("Final Squad Size", f"{summary['total_players']} players (Target: {target_squad_size})")
                        
                        st.info(f"**Built for Formation:** {st.session_state['selected_formation_key']}")
                        
                        st.markdown("**Actual Positional Breakdown:**")
                        pos_order_display = ['GK', 'DEF', 'MID', 'FWD']
                        for pos_cat in pos_order_display:
                            count = summary['position_counts'].get(pos_cat, 0)
                            min_req = strategist.squad_minimums.get(pos_cat,0)
                            st.write(f"• **{pos_cat}:** {count} (Min: {min_req})")
                    else:
                        st.warning("Squad could not be fully generated with current settings/budget.")
                
                # --- Full Player List & MRBs Table ---
                st.markdown('<hr><h2 class="section-header">📋 Full Player Database & Calculated Values</h2>', unsafe_allow_html=True)
                
                df_display_full_list = st.session_state['df_for_display'].copy()
                # Define all columns including raw and normalized KPIs for user inspection
                all_stats_cols = [
                    'Joueur', 'Club', 'simplified_position', 'Poste', 'Indispo ?', 'Cote', 'pvs', 'mrb', 'value_per_cost',
                    'recent_avg_rating', 'season_avg_rating', '%Titu', 'recent_goals', 'season_goals', 'recent_games_played_count',
                    'norm_recent_avg', 'norm_season_avg', 'norm_regularity', 'norm_recent_goals', 'norm_season_goals'
                ]
                display_all_stats_cols_exist = [col for col in all_stats_cols if col in df_display_full_list.columns]
                df_display_full_list = df_display_full_list[display_all_stats_cols_exist]

                df_display_full_list.rename(columns={
                    'Joueur': 'Player', 'simplified_position': 'Simp.Pos', 'Poste':'Orig.Pos', 'Indispo ?': 'Unavailable',
                    'Cote': 'Listed Price', 'pvs': 'PVS (0-100)', 'mrb': 'Calc. MRB', 'value_per_cost': 'PVS/MRB Ratio',
                    'recent_avg_rating': 'Rec.Avg.R (0-10)', 'season_avg_rating': 'Sea.Avg.R (0-10)',
                    '%Titu': 'Reg.%', 'recent_goals': 'Rec.G (N)', 'season_goals': 'Sea.G',
                    'recent_games_played_count': 'Rec.Games Plyd(N)',
                    'norm_recent_avg': 'N.Rec.Avg (0-100)', 'norm_season_avg': 'N.Sea.Avg (0-100)',
                    'norm_regularity': 'N.Reg.% (0-100)', 'norm_recent_goals': 'N.Rec.G (0-100)', 'norm_season_goals': 'N.Sea.G (0-100)'
                }, inplace=True)
                
                # Round numeric columns for display
                cols_to_round_full = [col for col in df_display_full_list.columns if 'Price' in col or 'PVS' in col or 'MRB' in col or 'Ratio' in col or 'Avg' in col or 'Reg.%' in col or 'Goals' in col or 'Plyd' in col]
                for col in cols_to_round_full:
                     if col in df_display_full_list.columns: # Check if column exists after rename
                        df_display_full_list[col] = pd.to_numeric(df_display_full_list[col], errors='coerce').round(2)


                search_term_all = st.text_input("🔍 Search All Players (name, club, position):", key="search_all_players_input")
                if search_term_all:
                    search_mask_all = df_display_full_list.apply(lambda row: row.astype(str).str.contains(search_term_all, case=False, na=False).any(), axis=1)
                    df_display_full_list = df_display_full_list[search_mask_all]
                
                st.dataframe(df_display_full_list.sort_values(by='PVS (0-100)', ascending=False), use_container_width=True, hide_index=True, height=600)

                csv_export_all = df_display_full_list.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Full Player Analysis (CSV)",
                    data=csv_export_all,
                    file_name="mpg_full_player_analysis_with_all_stats.csv",
                    mime="text/csv",
                    key="download_all_player_analysis"
                )
            
            elif 'df_for_display' not in st.session_state and uploaded_file: # Only show if file uploaded but no calculation run
                 st.info("📊 Configure your settings in the sidebar and click 'Calculate Optimal Squad & MRBs' to view results.")


        except pd.errors.ParserError as pe:
            st.error(f"⚠️ Error parsing the file: {str(pe)}. Please ensure it's a valid CSV or Excel file.")
        except ValueError as ve:
            st.error(f"⚠️ Error converting data: {str(ve)}. Check that numeric columns (like Cote, %Titu, gameweek ratings) contain valid numbers.")
        except Exception as e: # Catch-all for other unexpected errors
            st.error(f"⚠️ An unexpected error occurred: {str(e)}")
            st.exception(e) # Shows full traceback
            st.warning("Please check your file format against the guide and ensure all player data is sensible.")
    else:
        st.info("👈 Please upload your MPG ratings file using the sidebar to begin.")
        # Display Expected File Format Guide
        st.markdown('<hr><h2 class="section-header">📋 Expected File Format Guide</h2>', unsafe_allow_html=True)
        example_data = {
            'Joueur': ['Player A', 'Player B', 'Player C (Unavailable)', 'Player D'],
            'Poste': ['A', 'M', 'D', 'G'],
            'Club': ['Club X', 'Club Y', 'Club X', 'Club Z'],
            'Indispo ?': ['', '', 'TRUE', ''], 
            'Cote': [45, 30, 15, 10],
            '%Titu': [90, 75, 80, 95],
            'D34': ['7.5*', '6.5', '(5.0)', '7.0'], 
            'D33': ['(6.0)**', '7.0*', '6.0', '0'], 
            'D32': ['', '5.5', '4.5*', '(6.5)'],
            # ... other gameweek columns D31 down to D1
        }
        st.dataframe(pd.DataFrame(example_data), use_container_width=True, hide_index=True)
        st.markdown("""
        **Key Column Explanations:**
        - **Joueur**: Player's full name.
        - **Poste**: Original position (G, D, DL, DC, M, MD, MO, A). Simplified by app.
        - **Club**: Player's club.
        - **Indispo ?**: Availability. 'TRUE', 'OUI', '1', 'YES', 'VRAI' (case-insensitive) mark player as unavailable. Blank or other values mean available.
        - **Cote**: MPG Price (numeric). App ensures it's at least 1.
        - **%Titu**: Titularisation percentage (numeric, e.g., 75 for 75%).
        - **Dxx (e.g., D34...D1)**: Gameweek columns.
            - Format: Rating (e.g., `6.5`), `(SubRating)` (e.g., `(5.0)`), `Rating*` (1 goal), `Rating**` (2 goals).
            - Blank or '0' cell = Did Not Play (DNP) for that week.
            - Name columns `D<number>` (e.g., D1, D34). App sorts them to find most recent (D34 > D1). `D-34` etc. ignored.
        """)

if __name__ == "__main__":
    main()
