import streamlit as st

st.set_page_config(
    page_title="MPG Strategist Hub",
    page_icon="🔗",
    layout="wide"
)

st.title("🚀 MPG Auction Strategist Hub")
st.markdown("---")

st.header("Choose Your Strategist Tool:")

# --- Replace with your actual URLs ---
URL_HISTORICAL_APP = "https://mercatompg.streamlit.app"
URL_NEW_SEASON_APP = "https://newmercatompg.streamlit.app"

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Historical Data App")
    st.markdown("For leagues with past season performance data.")
    if URL_HISTORICAL_APP != "YOUR_URL_FOR_THE_HISTORICAL_DATA_APP_HERE":
        st.link_button("Launch Historical App", URL_HISTORICAL_APP)
    else:
        st.warning("Historical App URL not configured.")

with col2:
    st.subheader("🎯 New Season App")
    st.markdown("For new leagues, using subjective estimations.")
    if URL_NEW_SEASON_APP != "YOUR_URL_FOR_THE_NEW_SEASON_APP_HERE":
        st.link_button("Launch New Season App", URL_NEW_SEASON_APP)
    else:
        st.warning("New Season App URL not configured.")

st.sidebar.title("About")
st.sidebar.info(
    "This Hub provides access to different MPG Auction Strategist tools. "
    "Select an application above to open it."
)
