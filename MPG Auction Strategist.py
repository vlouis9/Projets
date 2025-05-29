# launcher.py
import streamlit as st

# Configure the page at the top of this main file.
st.set_page_config(
    page_title="MPG Auction Strategist Launcher",
    page_icon="🏆",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("MPG Auction Strategist Launcher")
st.write("Select an app to open:")

# If you know the URLs of your deployed apps, you can simply link to them.
# Replace the example URLs below with your actual deployed app URLs.

historical_url = "https://mercatompg.streamlit.app"  # URL for the historical_app
new_season_url   = "newmercatompg.streamlit.app"     # URL for the new_season_app

st.markdown(f"""  
### [Historical Season Data Mode]({historical_url})  
Click this link to open the Historical data app in a new tab.
""", unsafe_allow_html=True)

st.markdown(f"""  
### [New Season Mode (No Historical Data)]({new_season_url})  
Click this link to open the New Season app in a new tab.
""", unsafe_allow_html=True)

# Alternatively, you can provide buttons that open the apps via a simple HTML snippet.
def open_in_new_tab(url: str):
    # This JavaScript snippet opens the URL in a new tab.
    js = f"window.open('{url}')"  
    html = f'<input type="button" value="Open App" onclick="{js}">'
    st.markdown(html, unsafe_allow_html=True)

st.write("Or, use the buttons below:")

st.write("Historical Season Data Mode:")
open_in_new_tab(historical_url)

st.write("New Season Mode (No Historical Data):")
open_in_new_tab(new_season_url)
