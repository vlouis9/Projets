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

import streamlit as st
import pandas as pd

# =============================================================================
# Page Configuration (must be the very first Streamlit command)
# =============================================================================
st.set_page_config(
    page_title="New Season Mode - Step 1: Data Ingestion & Preprocessing",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("Step 1: Data Ingestion & Preprocessing")

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
