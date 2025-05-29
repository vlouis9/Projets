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
# 1. File Upload and Data Loading
# =============================================================================
uploaded_file = st.sidebar.file_uploader(
    "Upload New Season Data File (CSV/Excel)",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith('.csv'):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Error reading the file: {e}")
        st.stop()
else:
    st.info("Please upload your new season data file in the sidebar.")
    st.stop()

st.subheader("Raw Data Preview")
st.dataframe(df.head())

# =============================================================================
# 2. Sidebar Parameters for Composite Score Calculation
# =============================================================================
st.sidebar.markdown("### Composite Score Parameter Settings")
base_multiplier = st.sidebar.slider(
    "Cote Weight", 0.1, 5.0, 1.0, 0.1,
    help="Weight for the player's current rating (Cote)."
)
talent_weight = st.sidebar.slider(
    "Talent Potential Weight", 0.1, 5.0, 1.0, 0.1,
    help="Weight for the projected talent potential (0-10 scale)."
)
buzz_weight = st.sidebar.slider(
    "Market Buzz Weight", 0.1, 5.0, 1.0, 0.1,
    help="Weight for market buzz / hype (0-10 scale)."
)
expert_weight = st.sidebar.slider(
    "Expert Sentiment Weight", 0.1, 5.0, 1.0, 0.1,
    help="Weight for expert opinions (0-10 scale)."
)

st.sidebar.markdown("### Default Parameter Values (if not provided in file)")
default_talent = st.sidebar.number_input(
    "Default Talent Potential (0-10)", min_value=0.0, max_value=10.0, value=5.0,
    help="Use this if the file does not include talent ratings."
)
default_buzz = st.sidebar.number_input(
    "Default Market Buzz (0-10)", min_value=0.0, max_value=10.0, value=5.0,
    help="Use this if the file does not include buzz ratings."
)
default_expert = st.sidebar.number_input(
    "Default Expert Sentiment (0-10)", min_value=0.0, max_value=10.0, value=5.0,
    help="Use this if the file does not include expert ratings."
)

# =============================================================================
# 3. Data Preprocessing and Parameter Filling
# =============================================================================
# Ensure that the essential column "Cote" is numeric.
df['Cote'] = pd.to_numeric(df['Cote'], errors='coerce').fillna(1)

# If the extra parameters do not exist in the dataframe, add them as defaults.
if "talent_potential" not in df.columns:
    df["talent_potential"] = default_talent
if "market_buzz" not in df.columns:
    df["market_buzz"] = default_buzz
if "expert_sentiment" not in df.columns:
    df["expert_sentiment"] = default_expert

# =============================================================================
# 4. Composite Score Calculation
# =============================================================================
# In this example, we assume:
# - "Cote" is on a 0–100 scale,
# - The extra parameters are entered on a 0–10 scale and will be scaled by 10.
df['norm_cote'] = df['Cote']          # Assume cote is already normalized (0-100)
df['norm_talent'] = df['talent_potential'].astype(float) * 10
df['norm_buzz'] = df['market_buzz'].astype(float) * 10
df['norm_expert'] = df['expert_sentiment'].astype(float) * 10

# Total weight for combining the parameters:
total_weight = base_multiplier + talent_weight + buzz_weight + expert_weight

# Calculate composite player score "pvs" as the weighted average.
df['pvs'] = (
    df['norm_cote'] * base_multiplier +
    df['norm_talent'] * talent_weight +
    df['norm_buzz'] * buzz_weight +
    df['norm_expert'] * expert_weight
) / total_weight

# Clip the resulting score between 0 and 100:
df['pvs'] = df['pvs'].clip(0, 100)

# For continuity with your existing pipeline, we define MRB as a function of cote.
df['mrb'] = df['Cote']  # You can later modify this if desired.
df['value_per_cost'] = df['pvs'] / df['mrb'].replace(0, np.nan)
df['value_per_cost'].fillna(0, inplace=True)

# =============================================================================
# 5. Display the Evaluated Players
# =============================================================================
st.subheader("Evaluated Players")
st.dataframe(df.head())

# (You can add further functionality such as squad selection below.)
