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
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
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
</style>
""", unsafe_allow_html=True)

class MPGAuctionStrategist:
    def __init__(self):
        self.formations = { #
            "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3}, #
            "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2}, #
            "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3}, #
            "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2}, #
            "4-5-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1}, #
            "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2}, #
            "5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1}  #
        }
        self.squad_minimums = {"GK": 2, "DEF": 6, "MID": 6, "FWD": 4} #
        self.budget = 500

    def simplify_position(self, position: str) -> str:
        if pd.isna(position) or str(position).strip() == '':
            return 'UNKNOWN' #
        pos = str(position).upper().strip()
        if pos == 'G':
            return 'GK' #
        elif pos in ['D', 'DL', 'DC']: # Added 'D' for general defenders
            return 'DEF' #
        elif pos in ['M', 'MD', 'MO']: # Added 'M' for general midfielders
            return 'MID' #
        elif pos == 'A':
            return 'FWD' #
        else:
            return 'UNKNOWN' #

    def create_player_id(self, row) -> str:
        name = str(row.get('Joueur', '')).strip() #
        # Use the original 'Poste' for ID to ensure uniqueness if simplify_position changes,
        # but for actual logic, simplified_position is used.
        # For consistency in ID, let's use the simplified position
        simplified_pos = self.simplify_position(row.get('Poste', '')) #
        club = str(row.get('Club', '')).strip() #
        return f"{name}_{simplified_pos}_{club}"

    def extract_rating_and_goals(self, rating_str) -> Tuple[Optional[float], int, bool]:
        """Extract MPG rating, goals, and if played (not DNP)"""
        if pd.isna(rating_str) or str(rating_str).strip() == '' or str(rating_str).strip() == '0':
            return None, 0, False # DNP if blank or '0'
        
        rating_val_str = str(rating_str).strip()
        
        goals = rating_val_str.count('*') #
        
        # Remove parentheses and asterisks to get the rating number
        clean_rating_str = re.sub(r'[()\*]', '', rating_val_str) #
        
        try:
            rating = float(clean_rating_str)
            if rating == 0: # If after cleaning, rating is 0, treat as DNP as per user clarification.
                 return None, goals, False # This '0' might be a specific DNP marker or extremely poor play
            return rating, goals, True # Played
        except ValueError:
            return None, 0, False # If conversion fails, treat as DNP

    def get_gameweek_columns(self, df_columns: List[str]) -> List[str]:
        """Identify and sort gameweek columns (e.g., D1, D2, ..., D34 with D34 as most recent)"""
        gw_cols = []
        for col in df_columns:
            match = re.fullmatch(r'D(\d+)', col) # Matches D<number>
            if match:
                gw_cols.append({'name': col, 'number': int(match.group(1))})
        
        # Sort by gameweek number (descending for most recent first, e.g. D34, D33, ...)
        # This matches the assumption that user file might be D34, D33 ... D1
        # If file is D1, D2 ... D34, then sort ascending and pick from end for recent.
        # User said: "D1, D2, D3... In reverse order in the file." This implies D1 is most recent column if columns are sorted D1...D34.
        # However, D34 is typically the latest gameweek.
        # Let's assume standard notation: D34 is later than D1. We want the N latest gameweeks.
        # So we sort D1...D34 (ascending number) and then take the last N.
        sorted_gw_cols = sorted(gw_cols, key=lambda x: x['number'])
        return [col['name'] for col in sorted_gw_cols]


    def calculate_kpis(self, df: pd.DataFrame, n_recent: int) -> pd.DataFrame:
        result_df = df.copy()
        
        all_df_gameweek_cols = self.get_gameweek_columns(df.columns) # Correctly sorted, D1 to D34
        
        kpi_cols_to_init = [
            'recent_avg_rating', 'recent_goals', 'season_avg_rating', 'season_goals',
            'recent_games_played_count' # New KPI for filtering
        ]
        for col in kpi_cols_to_init: #
            result_df[col] = 0.0 #
            if 'count' in col: result_df[col] = 0


        for idx, row in result_df.iterrows():
            season_ratings_played = []
            season_goals_total = 0
            
            # Process all gameweeks for season stats
            for gw_col_name in all_df_gameweek_cols:
                rating, goals, played_this_gw = self.extract_rating_and_goals(row[gw_col_name])
                if played_this_gw and rating is not None: # Only consider if played and rating is valid
                    season_ratings_played.append(rating)
                    season_goals_total += goals
            
            result_df.at[idx, 'season_avg_rating'] = np.mean(season_ratings_played) if season_ratings_played else 0
            result_df.at[idx, 'season_goals'] = season_goals_total

            # Process N most recent gameweeks for recent form (user wants avg of played games)
            # The N most recent gameweeks are the last N columns in `all_df_gameweek_cols`
            recent_calendar_gws = all_df_gameweek_cols[-n_recent:] if len(all_df_gameweek_cols) >= n_recent else all_df_gameweek_cols
            
            recent_ratings_played = []
            recent_goals_scored_in_played_games = 0
            recent_games_played_count_in_window = 0

            for gw_col_name in recent_calendar_gws:
                rating, goals, played_this_gw = self.extract_rating_and_goals(row[gw_col_name])
                if played_this_gw and rating is not None: # Only consider if played and rating is valid
                    recent_ratings_played.append(rating)
                    recent_goals_scored_in_played_games += goals
                    recent_games_played_count_in_window += 1
            
            result_df.at[idx, 'recent_avg_rating'] = np.mean(recent_ratings_played) if recent_ratings_played else 0
            result_df.at[idx, 'recent_goals'] = recent_goals_scored_in_played_games
            result_df.at[idx, 'recent_games_played_count'] = recent_games_played_count_in_window
        
        return result_df

    def normalize_kpis(self, df: pd.DataFrame) -> pd.DataFrame:
        result_df = df.copy()
        
        result_df['norm_recent_avg'] = np.clip(result_df['recent_avg_rating'] * 10, 0, 100) #
        result_df['norm_season_avg'] = np.clip(result_df['season_avg_rating'] * 10, 0, 100) #
        
        # Ensure '%Titu' is numeric, coercing errors and filling NaNs with 0
        result_df['norm_regularity'] = pd.to_numeric(result_df['%Titu'], errors='coerce').fillna(0).clip(0, 100) #
        
        result_df['norm_recent_goals'] = 0.0 # Initialize for all
        result_df['norm_season_goals'] = 0.0 # Initialize for all

        for pos in ['MID', 'FWD']:
            pos_mask = result_df['simplified_position'] == pos #
            if pos_mask.sum() > 0:
                result_df.loc[pos_mask, 'norm_recent_goals'] = np.clip(result_df.loc[pos_mask, 'recent_goals'] * 20, 0, 100) #
                
                max_season_goals = result_df.loc[pos_mask, 'season_goals'].max()
                if max_season_goals > 0:
                    result_df.loc[pos_mask, 'norm_season_goals'] = np.clip(
                        (result_df.loc[pos_mask, 'season_goals'] / max_season_goals * 100), 0, 100 #
                    )
                else:
                    result_df.loc[pos_mask, 'norm_season_goals'] = 0 #
        
        return result_df

    def calculate_pvs(self, df: pd.DataFrame, weights: Dict[str, Dict[str, float]]) -> pd.DataFrame:
        result_df = df.copy()
        result_df['pvs'] = 0.0
        
        for pos_simplified in ['GK', 'DEF', 'MID', 'FWD']:
            pos_mask = result_df['simplified_position'] == pos_simplified #
            if not pos_mask.any():
                continue
            
            pos_weights = weights[pos_simplified]
            
            # Initialize PVS components to 0 to avoid errors if a norm_kpi is missing
            # (though normalize_kpis should create them all)
            pvs_calc = pd.Series(0.0, index=result_df.loc[pos_mask].index)
            
            pvs_calc += result_df.loc[pos_mask, 'norm_recent_avg'].fillna(0) * pos_weights.get('recent_avg', 0)
            pvs_calc += result_df.loc[pos_mask, 'norm_season_avg'].fillna(0) * pos_weights.get('season_avg', 0)
            pvs_calc += result_df.loc[pos_mask, 'norm_regularity'].fillna(0) * pos_weights.get('regularity', 0)
            
            if pos_simplified in ['MID', 'FWD']:
                pvs_calc += result_df.loc[pos_mask, 'norm_recent_goals'].fillna(0) * pos_weights.get('recent_goals', 0)
                pvs_calc += result_df.loc[pos_mask, 'norm_season_goals'].fillna(0) * pos_weights.get('season_goals', 0)
            
            result_df.loc[pos_mask, 'pvs'] = pvs_calc # PVS is now 0-100 scale
                                                    # (assuming weights sum to 1 and KPIs are 0-100)
                                                    # Or sum of (0-100 KPI * 0-1 weight)
        return result_df

    def calculate_mrb(self, df: pd.DataFrame, mrb_params: Dict) -> pd.DataFrame:
        result_df = df.copy() #
        
        # These params will now expect PVS on a 0-100 scale
        baseline_pvs = mrb_params['baseline_pvs_0_100'] 
        max_markup_pct_val = mrb_params['max_markup_pct'] / 100.0 # e.g., 50% -> 0.5
        points_for_max_markup_pvs = mrb_params['points_for_max_markup_0_100']
        absolute_max_bid_val = mrb_params['absolute_max_bid']
        
        def calc_mrb_dynamic(row):
            cote = row['Cote']
            pvs = row['pvs'] # This PVS is now 0-100
            
            # Ensure cote is numeric and not NaN, default to a high number if problematic for MRB logic
            if pd.isna(cote) or cote <= 0: cote = 1 # Min bid usually 1
            
            if pvs <= baseline_pvs:
                return float(cote) #
            
            excess_pvs = pvs - baseline_pvs
            
            # Ensure points_for_max_markup_pvs is not zero to avoid division error
            if points_for_max_markup_pvs == 0:
                 actual_markup_percentage = max_markup_pct_val # Max markup if any excess PVS
            else:
                actual_markup_percentage = min(max_markup_pct_val, (excess_pvs / points_for_max_markup_pvs) * max_markup_pct_val) #
            
            mrb = cote * (1 + actual_markup_percentage)
            mrb = min(mrb, absolute_max_bid_val)
            return round(float(mrb)) # Return as float/int, often bids are integers

        result_df['mrb'] = result_df.apply(calc_mrb_dynamic, axis=1)
        
        # Handle MRB being zero or NaN for ValuePerCost calculation
        safe_mrb = result_df['mrb'].replace(0, np.nan) # Avoid division by zero if MRB is 0
        result_df['value_per_cost'] = result_df['pvs'] / safe_mrb 
        result_df['value_per_cost'].fillna(0, inplace=True) # if MRB was 0 or NaN, VpC is 0
        
        return result_df

    def select_squad(self, df: pd.DataFrame, formation_key: str, target_squad_size: int, 
                     min_recent_games: int) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
        
        # Apply minimum recent games played filter
        if min_recent_games > 0:
            eligible_df = df[df['recent_games_played_count'] >= min_recent_games].copy()
        else:
            eligible_df = df.copy()

        # Filter out unavailable players
        eligible_df = eligible_df[~eligible_df['Indispo ?'].astype(str).str.upper().isin(['TRUE', 'OUI', '1'])].copy() #

        if eligible_df.empty:
            st.warning("No eligible players available after filtering. Adjust filters or check data.")
            return None, None

        selected_player_ids = []
        current_budget_spent = 0
        squad_player_counts_map = {pos: 0 for pos in self.squad_minimums.keys()} # e.g. {"GK":0, ...}
        
        # Ensure player_id is unique and available for selection
        eligible_df = eligible_df.drop_duplicates(subset=['player_id'])

        # --- Phase 1: Select Starters for Preferred Formation ---
        starters_for_formation = self.formations[formation_key].copy() # {pos: count} e.g. {"GK":1, "DEF":4,...}
        
        st.write("--- Phase 1: Selecting Starters ---") # Debug
        starters_selected_this_phase = []

        for pos, num_starters_needed in starters_for_formation.items():
            candidates = eligible_df[
                (eligible_df['simplified_position'] == pos) &
                (~eligible_df['player_id'].isin(selected_player_ids))
            ].sort_values(by='pvs', ascending=False) # Prioritize PVS for starters
            
            added_for_pos = 0
            for _, player_row in candidates.iterrows():
                if added_for_pos >= num_starters_needed:
                    break
                if current_budget_spent + player_row['mrb'] <= self.budget: #
                    player_data = player_row.to_dict()
                    player_data['is_starter'] = True
                    starters_selected_this_phase.append(player_data)
                    selected_player_ids.append(player_row['player_id'])
                    current_budget_spent += player_row['mrb']
                    squad_player_counts_map[pos] += 1
                    added_for_pos += 1
            # st.write(f"Selected {added_for_pos}/{num_starters_needed} starters for {pos}. Budget spent: {current_budget_spent}")


        # --- Phase 2: Fulfill Overall Squad Positional Minimums ---
        st.write("--- Phase 2: Fulfilling Squad Minimums ---") # Debug
        bench_selected_for_minimums = []

        for pos, overall_min_count in self.squad_minimums.items(): #
            needed_for_overall_min = max(0, overall_min_count - squad_player_counts_map[pos]) #
            if needed_for_overall_min == 0: #
                continue

            candidates = eligible_df[
                (eligible_df['simplified_position'] == pos) &
                (~eligible_df['player_id'].isin(selected_player_ids))
            ].sort_values(by='value_per_cost', ascending=False) # Value/Cost for bench
            
            added_for_pos_min = 0
            for _, player_row in candidates.iterrows():
                if added_for_pos_min >= needed_for_overall_min:
                    break
                if current_budget_spent + player_row['mrb'] <= self.budget: #
                    player_data = player_row.to_dict()
                    player_data['is_starter'] = False # These are for squad depth to meet minimums
                    bench_selected_for_minimums.append(player_data)
                    selected_player_ids.append(player_row['player_id'])
                    current_budget_spent += player_row['mrb']
                    squad_player_counts_map[pos] += 1
                    added_for_pos_min += 1
            # st.write(f"Selected {added_for_pos_min}/{needed_for_overall_min} for {pos} to meet minimums. Budget spent: {current_budget_spent}")


        # --- Phase 3: Complete the Squad to Total Squad Size ---
        st.write("--- Phase 3: Completing to Total Squad Size ---") # Debug
        final_bench_fill = []
        
        current_total_players = len(selected_player_ids)
        remaining_slots_to_fill_total = max(0, target_squad_size - current_total_players)

        if remaining_slots_to_fill_total > 0:
            candidates = eligible_df[
                (~eligible_df['player_id'].isin(selected_player_ids))
            ].sort_values(by='value_per_cost', ascending=False) # Value/Cost for remaining slots
            
            added_to_total = 0
            for _, player_row in candidates.iterrows():
                if added_to_total >= remaining_slots_to_fill_total:
                    break
                if current_budget_spent + player_row['mrb'] <= self.budget: #
                    player_data = player_row.to_dict()
                    player_data['is_starter'] = False
                    final_bench_fill.append(player_data)
                    selected_player_ids.append(player_row['player_id'])
                    current_budget_spent += player_row['mrb']
                    # Increment the count for the actual simplified position of the player
                    squad_player_counts_map[player_row['simplified_position']] += 1
                    added_to_total += 1
            # st.write(f"Selected {added_to_total}/{remaining_slots_to_fill_total} to reach total squad size. Budget spent: {current_budget_spent}")

        # Consolidate selected player data from the original dataframe
        # Add is_starter information correctly
        final_squad_df = eligible_df[eligible_df['player_id'].isin(selected_player_ids)].copy()
        
        # Create a mapping from player_id to their 'is_starter' status
        starter_status_map = {}
        for p_data in starters_selected_this_phase:
            starter_status_map[p_data['player_id']] = True
        for p_data in bench_selected_for_minimums:
            if p_data['player_id'] not in starter_status_map: # Don't override if already a starter
                 starter_status_map[p_data['player_id']] = False
        for p_data in final_bench_fill:
            if p_data['player_id'] not in starter_status_map:
                 starter_status_map[p_data['player_id']] = False
        
        final_squad_df['is_starter'] = final_squad_df['player_id'].map(starter_status_map).fillna(False)


        # Final summary calculation
        actual_total_cost = final_squad_df['mrb'].sum() # Sum MRB from the selected players in dataframe
        
        squad_summary = { #
            'total_players': len(final_squad_df), #
            'total_cost': actual_total_cost, #
            'remaining_budget': self.budget - actual_total_cost, #
            'position_counts': final_squad_df['simplified_position'].value_counts().to_dict() #
        }
        
        # Check if all minimums are met (important final validation)
        for pos, min_val in self.squad_minimums.items():
            if squad_summary['position_counts'].get(pos, 0) < min_val:
                st.warning(f"Warning: Could not meet minimum for {pos} ({squad_summary['position_counts'].get(pos, 0)}/{min_val}) with current budget/players.")
                # Potentially return None or an incomplete squad indicator if strict
        if len(final_squad_df) < target_squad_size and len(final_squad_df) < self.squad_minimums_sum_val: # squad_minimums_sum_val needs to be defined
             st.warning(f"Warning: Could not reach target squad size. Selected {len(final_squad_df)} players.")


        return final_squad_df, squad_summary
    
    @property # Helper property
    def squad_minimums_sum_val(self):
        return sum(self.squad_minimums.values())

def main():
    st.markdown('<h1 class="main-header">⚽ MPG Auction Strategist</h1>', unsafe_allow_html=True) #
    
    strategist = MPGAuctionStrategist()
    
    # --- Sidebar ---
    st.sidebar.markdown('<h2 class="section-header">📁 Data & Global Settings</h2>', unsafe_allow_html=True) #
    with st.sidebar.expander("File Upload & Data Settings", expanded=True):
        uploaded_file = st.sidebar.file_uploader(
            "Upload your MPG ratings file (CSV or Excel)",
            type=['csv', 'xlsx', 'xls'], #
            help="Ensure columns like 'Joueur', 'Poste', 'Club', 'Indispo ?', 'Cote', '%Titu', and gameweek columns (e.g., D1, D2... D34) are present."
        )
        n_recent = st.sidebar.number_input(
            "Number of Recent Games (N) for Form KPIs", #
            min_value=1, max_value=38, value=5, #
            help="Number of most recent calendar gameweeks to analyze for 'Recent Form'. Player's performance in *played* games within this window is averaged." # [cite: 52]
        )
        min_recent_games_played_filter = st.sidebar.number_input(
            "Filter: Min Games Played in Last N Calendar Weeks (0 to disable)",
            min_value=0, max_value=n_recent, value=0,
            help="Exclude players who actually played in fewer than this many games within the 'N Recent Games' window defined above. '0' means no filter."
        )

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                [cite_start]df_input = pd.read_csv(uploaded_file) #
            else:
                df_input = pd.read_excel(uploaded_file) #
            
            required_cols = ['Joueur', 'Poste', 'Club', 'Cote', '%Titu'] # 'Indispo ?' is optional but recommended
            missing_cols = [col for col in required_cols if col not in df_input.columns]
            if missing_cols: #
                st.error(f"Error: Missing required columns in uploaded file: {', '.join(missing_cols)}") #
                return

            # Initial data cleaning and preparation
            df_processed = df_input.copy()
            df_processed['simplified_position'] = df_processed['Poste'].apply(strategist.simplify_position) #
            df_processed['player_id'] = df_processed.apply(strategist.create_player_id, axis=1) #
            df_processed['Cote'] = pd.to_numeric(df_processed['Cote'], errors='coerce').fillna(1).clip(lower=1) # Ensure Cote is at least 1
            if 'Indispo ?' not in df_processed.columns:
                df_processed['Indispo ?'] = False # Assume available if column is missing
            else:
                df_processed['Indispo ?'] = df_processed['Indispo ?'].astype(str).str.upper().isin(['TRUE', 'OUI', '1', 'YES'])


            st.sidebar.success(f"✅ File '{uploaded_file.name}' loaded: {len(df_processed)} players found.")
            
            # --- UI Sections in Sidebar ---
            with st.sidebar.expander("👥 Squad Building Parameters", expanded=True): #
                st.markdown("Define your target squad structure.") # Help text example
                formation_key = st.selectbox(
                    "Preferred Starting Formation", #
                    options=list(strategist.formations.keys()), #
                    index=3,  # Default to 4-4-2
                    help="Select the primary starting formation you want to build towards."
                )
                target_squad_size = st.number_input(
                    "Total Squad Size", #
                    min_value=strategist.squad_minimums_sum_val, max_value=30, value=20, #
                    help=f"Target number of players in your full squad (Min: {strategist.squad_minimums_sum_val} based on 2GK,6D,6M,4F; Max: 30)."
                )

            with st.sidebar.expander("📊 KPI Weights", expanded=True): #
                st.markdown("Adjust how much each factor contributes to a player's Value Score (PVS) for their position. Weights are relative.") # Help text
                weights = {}
                for pos_simplified in ['GK', 'DEF', 'MID', 'FWD']:
                    st.markdown(f'<h4>{pos_simplified}</h4>', unsafe_allow_html=True) #
                    weights[pos_simplified] = { #
                        'recent_avg': st.slider(f"Recent Avg Rating", 0.0, 1.0, 0.30 if pos_simplified != 'GK' else 0.4, 0.01, key=f"{pos_simplified}_w_rec_avg", help="Weight for average MPG rating in recent N games (when played)."),
                        'season_avg': st.slider(f"Season Avg Rating", 0.0, 1.0, 0.30 if pos_simplified != 'GK' else 0.4, 0.01, key=f"{pos_simplified}_w_sea_avg", help="Weight for average MPG rating over the whole season (when played)."),
                        'regularity': st.slider(f"Regularity (%Titu)", 0.0, 1.0, 0.20 if pos_simplified != 'GK' else 0.2, 0.01, key=f"{pos_simplified}_w_reg", help="Weight for % of games started/played."),
                        'recent_goals': st.slider(f"Recent Goals", 0.0, 1.0, 0.10 if pos_simplified in ['MID', 'FWD'] else 0.0, 0.01, key=f"{pos_simplified}_w_rec_g", help="Weight for goals scored in recent N games (0 for GK/DEF)."),
                        'season_goals': st.slider(f"Season Goals", 0.0, 1.0, 0.10 if pos_simplified in ['MID', 'FWD'] else 0.0, 0.01, key=f"{pos_simplified}_w_sea_g", help="Weight for total season goals (0 for GK/DEF).")
                    }
            
            with st.sidebar.expander("💰 MRB Calculation Parameters", expanded=True): #
                st.markdown("Configure how 'Max Recommended Bid' (MRB) is calculated from Player Value Score (PVS) and listed Cote (Price). MRB is used as the 'cost' in squad selection.") # Help text
                mrb_params = {
                    'baseline_pvs_0_100': st.number_input("Baseline PVS (0-100 scale)", min_value=0, max_value=100, value=55, step=1, help="PVS a player needs to be considered 'fairly priced' at their Cote. MRB defaults to Cote if PVS is below this."),
                    'max_markup_pct': st.number_input("Max Markup % over Cote", min_value=0, max_value=200, value=30, step=5, help="Maximum percentage you're willing to bid over a player's Cote if their PVS is high."),
                    'points_for_max_markup_0_100': st.number_input("PVS Points (above baseline) for Max Markup", min_value=1, max_value=50, value=25, step=1, help="How many PVS points above baseline are needed to apply the full 'Max Markup %'."),
                    'absolute_max_bid': st.number_input("Absolute Max Bid for any Player", min_value=1, max_value=300, value=150, step=5, help="The absolute highest MRB the app will suggest for any single player.")
                }

            if st.sidebar.button("🚀 Calculate Optimal Squad & MRBs", type="primary", use_container_width=True): #
                with st.spinner("Crunching numbers... This might take a moment!"):
                    try:
                        df_kpis = strategist.calculate_kpis(df_processed, n_recent) #
                        df_norm_kpis = strategist.normalize_kpis(df_kpis) #
                        df_pvs = strategist.calculate_pvs(df_norm_kpis, weights) #
                        df_mrb = strategist.calculate_mrb(df_pvs, mrb_params) #
                        
                        squad_df_result, squad_summary_result = strategist.select_squad(df_mrb, formation_key, target_squad_size, min_recent_games_played_filter) #
                        
                        st.session_state['df_for_display'] = df_mrb # Save for full list display
                        st.session_state['squad_df_result'] = squad_df_result
                        st.session_state['squad_summary_result'] = squad_summary_result
                        st.session_state['selected_formation'] = formation_key
                        st.success("Calculation complete! Results below.")
                    except Exception as e:
                        st.error(f"An error occurred during calculation: {e}")
                        st.exception(e) # Shows full traceback for debugging

            # --- Main Panel for Results ---
            if 'squad_df_result' in st.session_state and st.session_state['squad_df_result'] is not None: #
                col1, col2 = st.columns([3, 1]) # Give more space to squad table

                with col1:
                    st.markdown('<h2 class="section-header">🏆 Suggested Squad</h2>', unsafe_allow_html=True) #
                    
                    squad_display_df = st.session_state['squad_df_result'].copy() #
                    # Define columns to show in suggested squad table
                    squad_cols_ordered = [
                        'Joueur', 'Club', 'simplified_position', 'is_starter', 
                        'mrb', 'Cote', 'pvs', 
                        'recent_avg_rating', 'season_avg_rating', '%Titu',
                        'recent_goals', 'season_goals',
                        'value_per_cost', 'player_id' # player_id for potential debugging or advanced use
                    ]
                    # Ensure all columns exist before trying to select them
                    squad_cols_ordered = [col for col in squad_cols_ordered if col in squad_display_df.columns]
                    squad_display_df = squad_display_df[squad_cols_ordered]

                    squad_display_df.rename(columns={ #
                        'Joueur': 'Player', 'simplified_position': 'Pos', 'is_starter': 'Starter',
                        'mrb': 'MRB (Cost)', 'Cote': 'Listed Price', 'pvs': 'PVS',
                        'recent_avg_rating': 'Rec.Avg.Rate', 'season_avg_rating': 'Sea.Avg.Rate',
                        '%Titu': 'Regularity %', 'recent_goals': 'Rec.Goals', 'season_goals': 'Sea.Goals',
                        'value_per_cost': 'Val/MRB'
                    }, inplace=True)

                    numeric_cols_squad = ['MRB (Cost)', 'Listed Price', 'PVS', 'Rec.Avg.Rate', 'Sea.Avg.Rate', 'Regularity %', 'Rec.Goals', 'Sea.Goals', 'Val/MRB'] #
                    for col in numeric_cols_squad:
                        if col in squad_display_df.columns:
                            squad_display_df[col] = pd.to_numeric(squad_display_df[col], errors='coerce').round(2)
                    
                    # Sort by starter, then position (GK, D, M, A), then PVS
                    pos_order = ['GK', 'DEF', 'MID', 'FWD']
                    squad_display_df['Pos'] = pd.Categorical(squad_display_df['Pos'], categories=pos_order, ordered=True)
                    squad_display_df = squad_display_df.sort_values(
                        by=['Starter', 'Pos', 'PVS'], 
                        ascending=[False, True, False] # Starters first, then by position, then best PVS
                    ) #
                    
                    st.dataframe(squad_display_df, use_container_width=True, hide_index=True) #

                with col2:
                    st.markdown('<h2 class="section-header">📈 Squad Summary</h2>', unsafe_allow_html=True) #
                    summary = st.session_state['squad_summary_result']
                    if summary:
                        st.metric("Total MRB Cost", f"€ {summary['total_cost']:.0f} / {strategist.budget}", delta_color="inverse") #
                        st.metric("Remaining Budget", f"€ {summary['remaining_budget']:.0f}") #
                        st.metric("Squad Size", f"{summary['total_players']} players") #
                        
                        st.info(f"**Target Formation:** {st.session_state['selected_formation']}") #
                        
                        st.markdown("**Actual Position Breakdown:**") #
                        for pos_cat in pos_order: # Display in logical order
                            count = summary['position_counts'].get(pos_cat, 0)
                            st.write(f"• {pos_cat}: {count}") #
                    else:
                        st.write("Squad summary not available.")
                
                # --- Full Player List & MRBs Table ---
                st.markdown('<hr><h2 class="section-header">📋 Full Player Database & Calculated Values</h2>', unsafe_allow_html=True) #
                
                df_display_full = st.session_state['df_for_display'].copy()
                # Define columns for full display
                full_list_cols_ordered = [
                    'Joueur', 'Club', 'simplified_position', 'Indispo ?', 'Cote', 'pvs', 'mrb', 'value_per_cost',
                    'recent_avg_rating', 'season_avg_rating', '%Titu',
                    'recent_goals', 'season_goals', 'recent_games_played_count',
                    'norm_recent_avg', 'norm_season_avg', 'norm_regularity', 
                    'norm_recent_goals', 'norm_season_goals'
                ]
                full_list_cols_ordered = [col for col in full_list_cols_ordered if col in df_display_full.columns]
                df_display_full = df_display_full[full_list_cols_ordered]


                df_display_full.rename(columns={ #
                    'Joueur': 'Player', 'simplified_position': 'Pos', 'Indispo ?': 'Unavailable',
                    'Cote': 'Listed Price', 'pvs': 'PVS', 'mrb': 'Calc. MRB', 'value_per_cost': 'Val/MRB',
                    'recent_avg_rating': 'Rec.Avg.Rate', 'season_avg_rating': 'Sea.Avg.Rate',
                    '%Titu': 'Reg.%', 'recent_goals': 'Rec.Goals', 'season_goals': 'Sea.Goals',
                    'recent_games_played_count': 'Rec.Games Plyd',
                    'norm_recent_avg': 'N.Rec.Avg', 'norm_season_avg': 'N.Sea.Avg',
                    'norm_regularity': 'N.Reg.%', 'norm_recent_goals': 'N.Rec.G', 'norm_season_goals': 'N.Sea.G'
                }, inplace=True)

                numeric_cols_full = ['Listed Price', 'PVS', 'Calc. MRB', 'Val/MRB', 
                                     'Rec.Avg.Rate', 'Sea.Avg.Rate', 'Reg.%', 
                                     'Rec.Goals', 'Sea.Goals', 'Rec.Games Plyd',
                                     'N.Rec.Avg', 'N.Sea.Avg', 'N.Reg.%',
                                     'N.Rec.G', 'N.Sea.G'] #
                for col in numeric_cols_full:
                    if col in df_display_full.columns:
                         df_display_full[col] = pd.to_numeric(df_display_full[col], errors='coerce').round(2)

                search_term = st.text_input("🔍 Search all players (name, club, position):", key="search_all_players") #
                if search_term:
                    search_mask = df_display_full.apply(lambda row: row.astype(str).str.contains(search_term, case=False, na=False).any(), axis=1) #
                    df_display_full = df_display_full[search_mask] #
                
                st.dataframe(df_display_full.sort_values(by='Val/MRB', ascending=False), use_container_width=True, hide_index=True) #

                # Download button
                csv_export = df_display_full.to_csv(index=False).encode('utf-8')
                st.download_button( #
                    label="📥 Download Full Player Analysis (CSV)", #
                    data=csv_export, #
                    file_name="mpg_full_player_analysis.csv", #
                    mime="text/csv", #
                    key="download_full_analysis"
                )
            
            elif 'df_for_display' not in st.session_state : # only show if no calculation has run yet
                 st.info("Configure settings and click 'Calculate Optimal Squad & MRBs' to view results.") #


        except Exception as e:
            st.error(f"⚠️ An error occurred with the file or processing: {str(e)}")
            st.exception(e) # Provides full traceback for easier debugging by the user
            st.warning("Please ensure your file format matches the expected structure and all player data is sensible.")

    else:
        st.info("👈 Please upload your MPG ratings file using the sidebar to begin.") #
        st.markdown('<h2 class="section-header">📋 Expected File Format Guide</h2>', unsafe_allow_html=True) #
        
        example_data = { #
            'Joueur': ['Player A', 'Player B', 'Player C (Unavailable)', 'Player D'],
            'Poste': ['A', 'M', 'D', 'G'], #
            'Club': ['Club X', 'Club Y', 'Club X', 'Club Z'], #
            'Indispo ?': ['', '', 'TRUE', ''], # Example for unavailability
            'Cote': [45, 30, 15, 10], #
            '%Titu': [90, 75, 80, 95], #
            'D34': ['7.5*', '6.5', '(5.0)', '7.0'], # Example gameweek data
            'D33': ['(6.0)**', '7.0*', '6.0', '0'], # '0' can mean DNP
            'D32': ['', '5.5', '4.5*', '(6.5)'], # Blank can mean DNP
            # ... other gameweek columns D31 down to D1
        }
        st.dataframe(pd.DataFrame(example_data), use_container_width=True, hide_index=True) #
        
        st.markdown("""
        **Key Column Descriptions:**
        - **Joueur**: Player's full name.
        - **Poste**: Original position (G, D, DL, DC, M, MD, MO, A). The app simplifies these to GK, DEF, MID, FWD.
        - **Club**: Player's current club.
        - **Indispo ?**: Player availability. Mark 'TRUE', 'OUI', or '1' if unavailable. Leave blank or use 'FALSE', 'NON', '0' if available. (Case-insensitive).
        - **Cote**: The player's listed MPG price/cost. Must be a number.
        - **%Titu**: Titularisation percentage (e.g., 75 for 75%). Must be a number.
        - **Dxx (e.g., D34, D33, ..., D1)**: Gameweek columns.
            - Ratings are numbers (e.g., 6.5).
            - Use `()` around rating if player was a substitute (e.g., `(5.0)`).
            - Use `*` for each goal scored (e.g., `7.5*` for 1 goal, `(6.0)**` for 2 goals as sub).
            - Blank or '0' in a gameweek cell is treated as Did Not Play (DNP) for that week.
            - Ensure gameweek columns are named `D<number>` (e.g. `D1`, `D2`...`D34`). The app will sort these numerically to determine recency (D34 is most recent). Columns like `D-34` will be ignored.
        """) #

if __name__ == "__main__":
    main()
