import streamlit as st
import pandas as pd
import numpy as np

# Set page configuration (must be the very first Streamlit command)
st.set_page_config(
    page_title="MPG Auction Strategist - New Season Mode",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("New Season Mode: Composite Player Evaluation")

# =============================================================================
# 1. File Upload
# =============================================================================
st.sidebar.markdown("### Upload Data File")
uploaded_file = st.sidebar.file_uploader(
    "Upload your new season data file (CSV/Excel)", 
    type=["csv", "xlsx", "xls"]
)

# If no file uploaded, stop further execution.
if uploaded_file is None:
    st.info("Please upload a data file using the sidebar.")
    st.stop()

# =============================================================================
# 2. Data Loading
# =============================================================================
try:
    if uploaded_file.name.endswith('.csv'):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)
except Exception as e:
    st.error(f"Error reading file: {e}")
    st.stop()

st.subheader("Raw Data Preview")
st.dataframe(df.head())

# =============================================================================
# 3. Preprocessing of the 'Cote' Column
# =============================================================================
# Convert the "Cote" column to numeric; if conversion fails, default to 1.
df['Cote'] = pd.to_numeric(df['Cote'], errors='coerce').fillna(1)

# Normalization Function
def normalize_player_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Normalizes player attributes to a consistent scale (0-100).
    """
    df_normalized = df.copy()

    df_normalized['Cote'] = pd.to_numeric(df_normalized['Cote'], errors='coerce').fillna(1).clip(1, 100)
    df_normalized['norm_cote'] = df_normalized['Cote']
    df_normalized['norm_talent'] = df_normalized.get('talent_potential', 5.0) * 10
    df_normalized['norm_buzz'] = df_normalized.get('market_buzz', 5.0) * 10
    df_normalized['norm_expert'] = df_normalized.get('expert_sentiment', 5.0) * 10
    df_normalized[['norm_talent', 'norm_buzz', 'norm_expert']] = df_normalized[['norm_talent', 'norm_buzz', 'norm_expert']].clip(0, 100)

    return df_normalized

def calculate_pvs(df: pd.DataFrame, base_multiplier: float, talent_weight: float, buzz_weight: float, expert_weight: float) -> pd.DataFrame:
    """
    Computes the Player Value Score (PVS) using a weighted formula.
    
    Parameters:
        df: DataFrame with normalized player attributes.
        base_multiplier: Weight for the 'Cote' (normalized baseline rating).
        talent_weight: Weight for the player's talent potential.
        buzz_weight: Weight for market buzz.
        expert_weight: Weight for expert sentiment.
        
    Returns:
        Updated DataFrame with a new column 'pvs'.
    """
    df_pvs = df.copy()

    # Ensure total weight sum is valid to avoid division errors
    total_weight = base_multiplier + talent_weight + buzz_weight + expert_weight
    if total_weight == 0:
        st.error("Invalid weight settings: total weight cannot be zero.")
        return df_pvs

    # Calculate PVS using a weighted average formula
    df_pvs['pvs'] = (
        (df_pvs['norm_cote'] * base_multiplier) +
        (df_pvs['norm_talent'] * talent_weight) +
        (df_pvs['norm_buzz'] * buzz_weight) +
        (df_pvs['norm_expert'] * expert_weight)
    ) / total_weight

    # Ensure PVS stays within the expected range (0-100)
    df_pvs['pvs'] = df_pvs['pvs'].clip(0, 100)

    return df_pvs

def calculate_mrb(df: pd.DataFrame, mrb_params_per_pos: Dict[str, Dict[str, float]]) -> pd.DataFrame:
    """
    Calculates Market Reference Bid (MRB) based on Player Value Score (PVS).
    
    Parameters:
        df: DataFrame containing evaluated players.
        mrb_params_per_pos: Dictionary of position-based MRB scaling factors.
        
    Returns:
        Updated DataFrame with 'mrb' and 'value_per_cost' columns.
    """
    df_mrb = df.copy()

    # Define MRB as baseline 'Cote' initially
    df_mrb['mrb'] = df_mrb['Cote']

    for pos, params in mrb_params_per_pos.items():
        mask = df_mrb['simplified_position'] == pos
        if not mask.any():
            continue

        max_bonus_factor = params.get('max_proportional_bonus_at_pvs100', 0.5)

        def compute_mrb(row):
            cote = row['Cote']
            pvs = row['pvs']
            pvs_scaled = pvs / 100.0
            bonus_factor = pvs_scaled * max_bonus_factor
            calculated_mrb = cote * (1 + bonus_factor)
            capped_mrb = min(calculated_mrb, cote * 2)  # Prevent excessive bids
            return int(round(max(cote, capped_mrb)))  # Ensure MRB is at least equal to Cote

        df_mrb.loc[mask, 'mrb'] = df_mrb.loc[mask].apply(compute_mrb, axis=1)

    # Ensure MRB remains an integer and calculate value-per-cost ratio
    df_mrb['mrb'] = df_mrb['mrb'].astype(int)
    df_mrb['value_per_cost'] = df_mrb['pvs'] / df_mrb['mrb'].replace(0, np.nan)
    df_mrb['value_per_cost'].fillna(0, inplace=True)

    return df_mrb
    
# =============================================================================
# 4. Sidebar Default Values for Extra Parameters
# =============================================================================
st.sidebar.markdown("### Default Values for Extra Parameters")
default_talent = st.sidebar.number_input(
    "Default Talent Potential (0-10)", 
    value=5.0, min_value=0.0, max_value=10.0, step=0.1,
    help="Used if the file does not include a 'talent_potential' column."
)
default_buzz = st.sidebar.number_input(
    "Default Market Buzz (0-10)", 
    value=5.0, min_value=0.0, max_value=10.0, step=0.1,
    help="Used if the file does not include a 'market_buzz' column."
)
default_expert = st.sidebar.number_input(
    "Default Expert Sentiment (0-10)", 
    value=5.0, min_value=0.0, max_value=10.0, step=0.1,
    help="Used if the file does not include an 'expert_sentiment' column."
)

# =============================================================================
# 5. Check for Extra Parameters and Fill Defaults if Needed
# =============================================================================
# For each extra parameter, verify if the column is present. If not, fill it with the default.
if "talent_potential" not in df.columns:
    st.warning("Column 'talent_potential' not found. Filling with default value.")
    df["talent_potential"] = default_talent

if "market_buzz" not in df.columns:
    st.warning("Column 'market_buzz' not found. Filling with default value.")
    df["market_buzz"] = default_buzz

if "expert_sentiment" not in df.columns:
    st.warning("Column 'expert_sentiment' not found. Filling with default value.")
    df["expert_sentiment"] = default_expert

# =============================================================================
# 6. Display Preprocessed Data Preview
# =============================================================================
st.subheader("Preprocessed Data Preview")
st.dataframe(df.head())

# Optionally, you may include additional sanity checks such as:
# - Verifying that "Cote" falls within an expected range.
# - Checking for any missing values in critical columns, etc.
