import streamlit as st

st.set_page_config(
    page_title="MPG Strategist Hub",
    page_icon="👋",
    layout="wide"
)

st.title("🚀 Welcome to the MPG Auction Strategist Hub!")
st.markdown("---")

st.subheader("Please select an application from the navigation sidebar to get started.")

st.markdown("""
    Use the sidebar on the left to choose between:

    * **Historical Data App**: For analyzing players and building squads when you have season performance data (scores, goals, regularity, etc.).

    * **New Season App**: Designed for new leagues or when detailed historical data is unavailable. This app uses your subjective estimations for player performance, potential, regularity, and goals.
    """)

st.info("👈 Click on the arrow in the top-left corner to open the sidebar if it's hidden.")

