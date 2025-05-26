    import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import io

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
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2e7d32;
        margin-top: 1.5rem;
        margin-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .starter-row {
        background-color: #e8f5e8 !important;
    }
</style>
""", unsafe_allow_html=True)

class MPGAuctionStrategist:
    def __init__(self):
        self.formations = {
            "3-4-3": {"GK": 1, "DEF": 3, "MID": 4, "FWD": 3},
            "3-5-2": {"GK": 1, "DEF": 3, "MID": 5, "FWD": 2},
            "4-3-3": {"GK": 1, "DEF": 4, "MID": 3, "FWD": 3},
            "4-4-2": {"GK": 1, "DEF": 4, "MID": 4, "FWD": 2},
            "4-5-1": {"GK": 1, "DEF": 4, "MID": 5, "FWD": 1},
            "5-3-2": {"GK": 1, "DEF": 5, "MID": 3, "FWD": 2},
            "5-4-1": {"GK": 1, "DEF": 5, "MID": 4, "FWD": 1}
        }
        
        self.squad_minimums = {"GK": 2, "DEF": 6, "MID": 6, "FWD": 4}
        self.budget = 500
        
    def simplify_position(self, position: str) -> str:
        """Simplify position to GK, DEF, MID, FWD"""
        pos = str(position).upper().strip()
        if pos == 'G':
            return 'GK'
        elif pos in ['DL', 'DC']:
            return 'DEF'
        elif pos in ['MD', 'MO']:
            return 'MID'
        elif pos == 'A':
            return 'FWD'
        else:
            return 'UNKNOWN'
    
    def create_player_id(self, row) -> str:
        """Create unique player ID from Name + Position + Club"""
        name = str(row.get('Joueur', '')).strip()
        position = self.simplify_position(row.get('Poste', ''))
        club = str(row.get('Club', '')).strip()
        return f"{name}_{position}_{club}"
    
    def extract_rating_and_goals(self, rating_str) -> Tuple[Optional[float], int]:
        """Extract MPG rating and goals from rating string"""
        if pd.isna(rating_str) or rating_str == '':
            return None, 0
        
        rating_str = str(rating_str).strip()
        
        # Count goals (asterisks)
        goals = rating_str.count('*')
        
        # Extract rating (remove parentheses and asterisks)
        clean_str = re.sub(r'[()\\*]', '', rating_str)
        
        try:
            rating = float(clean_str)
            return rating, goals
        except:
            return None, 0
    
    def calculate_kpis(self, df: pd.DataFrame, n_recent: int) -> pd.DataFrame:
        """Calculate all KPIs for players"""
        result_df = df.copy()
        
        # Get gameweek columns (assuming they're numeric columns after the main columns)
        main_cols = ['Joueur', 'Poste', 'Club', 'Indispo ?', 'Cote', '%Titu']
        gameweek_cols = [col for col in df.columns if col not in main_cols]
        
        # Initialize KPI columns
        kpi_cols = ['recent_avg_rating', 'recent_goals', 'season_avg_rating', 'season_goals']
        for col in kpi_cols:
            result_df[col] = 0.0
        
        for idx, row in result_df.iterrows():
            all_ratings = []
            all_goals = []
            recent_ratings = []
            recent_goals_count = 0
            
            # Process all gameweeks
            for col in gameweek_cols:
                rating, goals = self.extract_rating_and_goals(row[col])
                
                # For season stats
                if rating is not None:
                    all_ratings.append(rating)
                    all_goals.append(goals)
                else:
                    # DNP counts as 0 for season averages too
                    all_ratings.append(0)
                    all_goals.append(0)
            
            # Recent stats (last N gameweeks)
            if len(all_ratings) >= n_recent:
                recent_ratings = all_ratings[-n_recent:]
                recent_goals_count = sum(all_goals[-n_recent:])
            else:
                recent_ratings = all_ratings
                recent_goals_count = sum(all_goals)
            
            # Calculate KPIs
            result_df.at[idx, 'recent_avg_rating'] = np.mean(recent_ratings) if recent_ratings else 0
            result_df.at[idx, 'recent_goals'] = recent_goals_count
            result_df.at[idx, 'season_avg_rating'] = np.mean(all_ratings) if all_ratings else 0
            result_df.at[idx, 'season_goals'] = sum(all_goals)
        
        return result_df
    
    def normalize_kpis(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normalize KPIs to 0-100 scale"""
        result_df = df.copy()
        
        # Normalize ratings (multiply by 10, cap at 100)
        result_df['norm_recent_avg'] = np.clip(result_df['recent_avg_rating'] * 10, 0, 100)
        result_df['norm_season_avg'] = np.clip(result_df['season_avg_rating'] * 10, 0, 100)
        
        # Normalize regularity (%Titu is already 0-100)
        result_df['norm_regularity'] = pd.to_numeric(result_df['%Titu'], errors='coerce').fillna(0)
        
        # Normalize goals by position
        for pos in ['MID', 'FWD']:
            pos_mask = result_df['simplified_position'] == pos
            
            if pos_mask.sum() > 0:
                # Recent goals: 5+ = 100
                result_df.loc[pos_mask, 'norm_recent_goals'] = np.clip(
                    result_df.loc[pos_mask, 'recent_goals'] * 20, 0, 100
                )
                
                # Season goals: max scorer = 100
                max_season_goals = result_df.loc[pos_mask, 'season_goals'].max()
                if max_season_goals > 0:
                    result_df.loc[pos_mask, 'norm_season_goals'] = (
                        result_df.loc[pos_mask, 'season_goals'] / max_season_goals * 100
                    )
                else:
                    result_df.loc[pos_mask, 'norm_season_goals'] = 0
            else:
                result_df.loc[pos_mask, 'norm_recent_goals'] = 0
                result_df.loc[pos_mask, 'norm_season_goals'] = 0
        
        # For GK and DEF, goals are not applicable
        gk_def_mask = result_df['simplified_position'].isin(['GK', 'DEF'])
        result_df.loc[gk_def_mask, 'norm_recent_goals'] = 0
        result_df.loc[gk_def_mask, 'norm_season_goals'] = 0
        
        return result_df
    
    def calculate_pvs(self, df: pd.DataFrame, weights: Dict) -> pd.DataFrame:
        """Calculate Player Value Score"""
        result_df = df.copy()
        result_df['pvs'] = 0.0
        
        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            pos_mask = result_df['simplified_position'] == pos
            if pos_mask.sum() == 0:
                continue
                
            pos_weights = weights[pos]
            
            # Calculate weighted sum
            pvs_values = (
                result_df.loc[pos_mask, 'norm_recent_avg'] * pos_weights['recent_avg'] +
                result_df.loc[pos_mask, 'norm_season_avg'] * pos_weights['season_avg'] +
                result_df.loc[pos_mask, 'norm_regularity'] * pos_weights['regularity'] +
                result_df.loc[pos_mask, 'norm_recent_goals'] * pos_weights['recent_goals'] +
                result_df.loc[pos_mask, 'norm_season_goals'] * pos_weights['season_goals']
            ) / 100  # Normalize to reasonable scale
            
            result_df.loc[pos_mask, 'pvs'] = pvs_values
        
        return result_df
    
    def calculate_mrb(self, df: pd.DataFrame, mrb_params: Dict) -> pd.DataFrame:
        """Calculate Max Recommended Bid"""
        result_df = df.copy()
        
        baseline_pvs = mrb_params['baseline_pvs']
        max_markup = mrb_params['max_markup'] / 100
        points_for_max = mrb_params['points_for_max']
        absolute_max = mrb_params['absolute_max']
        
        def calc_mrb(row):
            cote = row['Cote']
            pvs = row['pvs']
            
            if pvs <= baseline_pvs:
                return cote
            
            # Calculate markup percentage
            excess_pvs = pvs - baseline_pvs
            markup_pct = min(max_markup, (excess_pvs / points_for_max) * max_markup)
            
            mrb = cote * (1 + markup_pct)
            return min(mrb, absolute_max)
        
        result_df['mrb'] = result_df.apply(calc_mrb, axis=1)
        result_df['value_per_cost'] = result_df['pvs'] / result_df['mrb']
        
        return result_df
    
    def select_squad(self, df: pd.DataFrame, formation: str, squad_size: int) -> Tuple[pd.DataFrame, Dict]:
        """Select optimal squad based on formation and constraints"""
        available_df = df[df['Indispo ?'] != 'TRUE'].copy()
        selected_players = []
        remaining_budget = self.budget
        
        # Phase 1: Select starters for formation
        formation_needs = self.formations[formation].copy()
        
        for pos, needed in formation_needs.items():
            pos_players = available_df[
                (available_df['simplified_position'] == pos) & 
                (~available_df['player_id'].isin([p['player_id'] for p in selected_players]))
            ].copy()
            
            # Sort by PVS (descending)
            pos_players = pos_players.sort_values('pvs', ascending=False)
            
            count = 0
            for _, player in pos_players.iterrows():
                if count >= needed or player['mrb'] > remaining_budget:
                    continue
                    
                selected_players.append({
                    'player_id': player['player_id'],
                    'position': pos,
                    'mrb': player['mrb'],
                    'is_starter': True
                })
                remaining_budget -= player['mrb']
                count += 1
        
        # Phase 2: Meet squad minimums
        current_counts = {pos: sum(1 for p in selected_players if p['position'] == pos) 
                         for pos in self.squad_minimums.keys()}
        
        for pos, minimum in self.squad_minimums.items():
            needed = max(0, minimum - current_counts[pos])
            if needed == 0:
                continue
                
            pos_players = available_df[
                (available_df['simplified_position'] == pos) & 
                (~available_df['player_id'].isin([p['player_id'] for p in selected_players]))
            ].copy()
            
            # Sort by value per cost (descending)
            pos_players = pos_players.sort_values('value_per_cost', ascending=False)
            
            count = 0
            for _, player in pos_players.iterrows():
                if count >= needed or player['mrb'] > remaining_budget:
                    continue
                    
                selected_players.append({
                    'player_id': player['player_id'],
                    'position': pos,
                    'mrb': player['mrb'],
                    'is_starter': False
                })
                remaining_budget -= player['mrb']
                count += 1
        
        # Phase 3: Fill remaining squad slots
        remaining_slots = squad_size - len(selected_players)
        if remaining_slots > 0:
            remaining_players = available_df[
                ~available_df['player_id'].isin([p['player_id'] for p in selected_players])
            ].copy()
            
            remaining_players = remaining_players.sort_values('value_per_cost', ascending=False)
            
            count = 0
            for _, player in remaining_players.iterrows():
                if count >= remaining_slots or player['mrb'] > remaining_budget:
                    continue
                    
                selected_players.append({
                    'player_id': player['player_id'],
                    'position': player['simplified_position'],
                    'mrb': player['mrb'],
                    'is_starter': False
                })
                remaining_budget -= player['mrb']
                count += 1
        
        # Create result dataframe
        selected_ids = [p['player_id'] for p in selected_players]
        starter_ids = [p['player_id'] for p in selected_players if p['is_starter']]
        
        squad_df = df[df['player_id'].isin(selected_ids)].copy()
        squad_df['is_starter'] = squad_df['player_id'].isin(starter_ids)
        
        # Calculate summary
        total_cost = sum(p['mrb'] for p in selected_players)
        summary = {
            'total_players': len(selected_players),
            'total_cost': total_cost,
            'remaining_budget': self.budget - total_cost,
            'position_counts': {pos: sum(1 for p in selected_players if p['position'] == pos) 
                              for pos in ['GK', 'DEF', 'MID', 'FWD']}
        }
        
        return squad_df, summary

def main():
    st.markdown('<h1 class="main-header">⚽ MPG Auction Strategist</h1>', unsafe_allow_html=True)
    
    # Initialize the strategist
    strategist = MPGAuctionStrategist()
    
    # Sidebar for inputs
    st.sidebar.markdown('<h2 class="section-header">📁 Data Upload</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.sidebar.file_uploader(
        "Upload your MPG ratings file",
        type=['csv', 'xlsx', 'xls'],
        help="Upload your Excel or CSV file containing player ratings data"
    )
    
    if uploaded_file is not None:
        # Load data
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.sidebar.success(f"✅ File loaded: {len(df)} players")
            
            # Process data
            df['simplified_position'] = df.apply(strategist.simplify_position, axis=1)
            df['player_id'] = df.apply(strategist.create_player_id, axis=1)
            df['Cote'] = pd.to_numeric(df['Cote'], errors='coerce').fillna(0)
            
            # Global Settings
            st.sidebar.markdown('<h2 class="section-header">⚙️ Global Settings</h2>', unsafe_allow_html=True)
            
            n_recent = st.sidebar.number_input(
                "Number of Recent Games (N)",
                min_value=1, max_value=20, value=5,
                help="Number of recent gameweeks to consider for recent form KPIs"
            )
            
            # MRB Parameters
            st.sidebar.markdown('<h3>💰 MRB Calculation Parameters</h3>', unsafe_allow_html=True)
            
            mrb_params = {
                'baseline_pvs': st.sidebar.number_input("Baseline PVS", min_value=0.0, max_value=10.0, value=3.0, step=0.1),
                'max_markup': st.sidebar.number_input("Max Markup %", min_value=0, max_value=200, value=50),
                'points_for_max': st.sidebar.number_input("Points for Max Markup", min_value=0.1, max_value=10.0, value=2.0, step=0.1),
                'absolute_max': st.sidebar.number_input("Absolute Max Bid", min_value=1, max_value=200, value=100)
            }
            
            # KPI Weights
            st.sidebar.markdown('<h2 class="section-header">📊 KPI Weights</h2>', unsafe_allow_html=True)
            
            weights = {}
            for pos in ['GK', 'DEF', 'MID', 'FWD']:
                st.sidebar.markdown(f'<h4>{pos}</h4>', unsafe_allow_html=True)
                
                weights[pos] = {
                    'recent_avg': st.sidebar.slider(f"{pos} - Recent Avg Rating", 0.0, 1.0, 0.3, 0.05, key=f"{pos}_recent_avg"),
                    'season_avg': st.sidebar.slider(f"{pos} - Season Avg Rating", 0.0, 1.0, 0.3, 0.05, key=f"{pos}_season_avg"),
                    'regularity': st.sidebar.slider(f"{pos} - Regularity (%Titu)", 0.0, 1.0, 0.2, 0.05, key=f"{pos}_regularity"),
                    'recent_goals': st.sidebar.slider(f"{pos} - Recent Goals", 0.0, 1.0, 0.1 if pos in ['MID', 'FWD'] else 0.0, 0.05, key=f"{pos}_recent_goals"),
                    'season_goals': st.sidebar.slider(f"{pos} - Season Goals", 0.0, 1.0, 0.1 if pos in ['MID', 'FWD'] else 0.0, 0.05, key=f"{pos}_season_goals")
                }
            
            # Squad Building Parameters
            st.sidebar.markdown('<h2 class="section-header">👥 Squad Building</h2>', unsafe_allow_html=True)
            
            formation = st.sidebar.selectbox(
                "Preferred Starting Formation",
                options=list(strategist.formations.keys()),
                index=2  # Default to 4-3-3
            )
            
            squad_size = st.sidebar.number_input(
                "Total Squad Size",
                min_value=18, max_value=30, value=22
            )
            
            # Calculate button
            if st.sidebar.button("🚀 Calculate Optimal Squad", type="primary"):
                with st.spinner("Calculating optimal squad..."):
                    # Process data
                    df_with_kpis = strategist.calculate_kpis(df, n_recent)
                    df_normalized = strategist.normalize_kpis(df_with_kpis)
                    df_with_pvs = strategist.calculate_pvs(df_normalized, weights)
                    df_final = strategist.calculate_mrb(df_with_pvs, mrb_params)
                    
                    # Select squad
                    squad_df, squad_summary = strategist.select_squad(df_final, formation, squad_size)
                    
                    # Store in session state
                    st.session_state['df_final'] = df_final
                    st.session_state['squad_df'] = squad_df
                    st.session_state['squad_summary'] = squad_summary
                    st.session_state['formation'] = formation
            
            # Main panel - Results
            if 'squad_df' in st.session_state:
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown('<h2 class="section-header">🏆 Suggested Squad</h2>', unsafe_allow_html=True)
                    
                    # Squad table
                    squad_display = st.session_state['squad_df'][
                        ['Joueur', 'Club', 'simplified_position', 'Cote', 'pvs', 'mrb', 
                         'recent_avg_rating', 'season_avg_rating', 'norm_regularity', 'is_starter']
                    ].copy()
                    
                    squad_display.columns = ['Player', 'Club', 'Position', 'Cote', 'PVS', 'MRB', 
                                           'Recent Avg', 'Season Avg', 'Regularity %', 'Starter']
                    
                    # Round numeric columns
                    numeric_cols = ['Cote', 'PVS', 'MRB', 'Recent Avg', 'Season Avg', 'Regularity %']
                    for col in numeric_cols:
                        squad_display[col] = squad_display[col].round(2)
                    
                    # Sort by starter status and PVS
                    squad_display = squad_display.sort_values(['Starter', 'PVS'], ascending=[False, False])
                    
                    st.dataframe(
                        squad_display,
                        use_container_width=True,
                        hide_index=True
                    )
                
                with col2:
                    st.markdown('<h2 class="section-header">📈 Squad Summary</h2>', unsafe_allow_html=True)
                    
                    summary = st.session_state['squad_summary']
                    
                    # Budget metrics
                    st.metric("Total Cost", f"€{summary['total_cost']:.0f}", f"€{summary['remaining_budget']:.0f} remaining")
                    st.metric("Squad Size", f"{summary['total_players']} players")
                    
                    # Formation info
                    st.info(f"**Formation:** {st.session_state['formation']}")
                    
                    # Position breakdown
                    st.markdown("**Position Breakdown:**")
                    for pos, count in summary['position_counts'].items():
                        st.write(f"• {pos}: {count}")
                
                # Full player list
                st.markdown('<h2 class="section-header">📋 Full Player List & MRBs</h2>', unsafe_allow_html=True)
                
                full_list = st.session_state['df_final'][
                    st.session_state['df_final']['Indispo ?'] != 'oui'
                ][['Joueur', 'Club', 'simplified_position', 'Cote', 'pvs', 'mrb', 'value_per_cost']].copy()
                
                full_list.columns = ['Player', 'Club', 'Position', 'Cote', 'PVS', 'MRB', 'Value/Cost']
                
                # Round numeric columns
                numeric_cols = ['Cote', 'PVS', 'MRB', 'Value/Cost']
                for col in numeric_cols:
                    full_list[col] = full_list[col].round(3)
                
                # Sort by value per cost
                full_list = full_list.sort_values('Value/Cost', ascending=False)
                
                # Add search functionality
                search_term = st.text_input("🔍 Search players:", placeholder="Enter player name, club, or position...")
                
                if search_term:
                    mask = full_list.apply(lambda x: x.astype(str).str.contains(search_term, case=False, na=False).any(), axis=1)
                    full_list = full_list[mask]
                
                st.dataframe(
                    full_list,
                    use_container_width=True,
                    hide_index=True
                )
                
                # Download button for results
                csv = full_list.to_csv(index=False)
                st.download_button(
                    label="📥 Download Full Player List",
                    data=csv,
                    file_name="mpg_player_analysis.csv",
                    mime="text/csv"
                )
            
            else:
                st.info("👆 Please configure your settings in the sidebar and click 'Calculate Optimal Squad' to see results!")
        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")
            st.info("Please make sure your file has the required columns: Joueur, Poste, Club, Indispo ?, Cote, %Titu")
    
    else:
        st.info("👈 Please upload your MPG ratings file to get started!")
        
        # Show example of expected format
        st.markdown('<h2 class="section-header">📋 Expected File Format</h2>', unsafe_allow_html=True)
        
        example_data = {
            'Joueur': ['Messi', 'Mbappé', 'Neymar'],
            'Poste': ['A', 'A', 'MO'],
            'Club': ['PSG', 'PSG', 'PSG'],
            'Indispo ?': ['', '', 'oui'],
            'Cote': [25, 30, 20],
            '%Titu': [95, 85, 60],
            'GW1': ['6.5*', '7.2*', ''],
            'GW2': ['(5.8)', '8.1**', '5.5'],
            'GW3': ['7.9', '(4.2)', '6.8*']
        }
        
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df, use_container_width=True)
        
        st.markdown("""
        **Column Descriptions:**
        - **Joueur**: Player name
        - **Poste**: Position (G, DL/DC, MD/MO, A)
        - **Club**: Player's club
        - **Indispo ?**: Availability ('TRUE' for unavailable)
        - **Cote**: MPG Price
        - **%Titu**: Titularisation percentage
        - **GW1, GW2, etc.**: Gameweek ratings (use * for goals, () for non-starters)
        """)

if __name__ == "__main__":
    main()
