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
URL_SANDBOX_APP = "https://newmpgseason.streamlit.app"

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Ongoing Season App")
    st.markdown("For leagues with season data.")
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
        st.warning("🏝️ New Season App URL not configured.")

with col3:
    st.subheader("🎯 SandBox")
    st.markdown("To play with data and players")
    if URL_SANDBOX_APP != "YOUR_URL_FOR_THE_SANDBOX_APP_HERE":
        st.link_button("Launch Sandbox App", URL_SANDBOX_APP)
    else:
        st.warning("Sandbox App URL not configured.")

st.sidebar.title("About")
st.sidebar.info(
    "This Hub provides access to different MPG Auction Strategist tools. "
    "Select an application above to open it."
)
