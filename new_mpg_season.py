import streamlit as st
import pandas as pd
import numpy as np
import re
import io
import json
import matplotlib.pyplot as plt
import seaborn as sns

# ---- PAGE CONFIG ----
st.set_page_config(
    page_title="MPG Manual Squad Builder",
    page_icon="🌟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---- CUSTOM CSS ----
st.markdown("""
<style>
    :root {
        --primary: #2563eb;
        --secondary: #10b981;
        --accent: #8b5cf6;
        --background: #f8fafc;
        --card: #ffffff;
        --text: #0f172a;
        --border: #e2e8f0;
    }
    body {
        background-color: var(--background);
        color: var(--text);
        font-family: 'Inter', sans-serif;
    }
    .main-header {
        font-size: 2.5rem; 
        font-weight: 800; 
        text-align: center; 
        margin-bottom: 1.5rem;
        padding-bottom: 0.5rem;
        border-bottom: 3px solid var(--secondary);
        background: linear-gradient(90deg, var(--primary), var(--accent));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 700;
        color: var(--primary);
        margin-top: 1.5rem;
        margin-bottom: 1rem;
        padding-left: 0.5rem;
        border-left: 4px solid var(--secondary);
    }
    .stButton>button {
        background: linear-gradient(135deg, var(--primary), var(--accent));
        color: white !important;
        font-weight: 600;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        border: none;
        width: 100%;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(37, 99, 235, 0.3);
    }
    .card {
        background-color: var(--card);
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        margin-bottom: 1.5rem;
        border: 1px solid var(--border);
    }
    .metric-card {
        background: linear-gradient(135deg, var(--card), var(--card));
        text-align: center;
        padding: 1rem;
        border-radius: 12px;
        border-left: 4px solid var(--primary);
    }
    .position-tag {
        display: inline-block;
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        color: var(--text);
    }
    .GK-tag { background: linear-gradient(135deg, #dbeafe, #93c5fd); }
    .DEF-tag { background: linear-gradient(135deg, #dcfce7, #86efac); }
    .MID-tag { background: linear-gradient(135deg, #fef3c7, #fcd34d); }
    .FWD-tag { background: linear-gradient(135deg, #fee2e2, #fca5a5); }
    .starter-badge {
        background-color: var(--secondary);
        color: white;
        border-radius: 4px;
        padding: 0.15rem 0.5rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .progress-bar {
        height: 8px;
        border-radius: 4px;
        background-color: var(--border);
        margin-top: 0.5rem;
        overflow: hidden;
    }
    .progress-fill {
        height: 100%;
        background: linear-gradient(90deg, var(--primary), var(--accent));
        border-radius: 4px;
    }
    .player-card {
        width: 120px;
        padding: 1rem;
        margin: 0 0.5rem;
        border-radius: 8px;
        text-align: center;
        background: var(--card);
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        border: 1px solid var(--border);
        transition: all 0.3s ease;
    }
    .player-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    .club-badge {
        width: 24px;
        height: 24px;
        display: inline-block;
        border-radius: 50%;
        background-color: var(--primary);
        color: white;
        font-size: 0.7rem;
        line-height: 24px;
        margin-right: 0.5rem;
    }
    .dataframe th {background-color: var(--border) !important;}
    .dataframe td {border-bottom: 1px solid var(--border);}
</style>
""", unsafe_allow_html=True)

# ---- CONSTANTS ----
CLUB_TIERS = {
    "Winner": 100,
    "European": 75,
    "Average": 50,
    "Relegation": 25
}
CLUB_TIERS_LABELS = list(CLUB_TIERS.keys())
NEW_PLAYER_SCORE_OPTIONS = [0, 25, 50, 75, 100]
DEFAULT_SQUAD_SIZE = 20

PREDEFINED_PROFILES = {
    "Custom": "custom",
    "Balanced Value": {
        "kpi_weights": {
            'GK': {'estimated_performance': 0.40, 'estimated_potential': 0.30, 'estimated_regularity': 0.30, 'estimated_goals': 0.0, 'team_ranking': 0.0},
            'DEF': {'estimated_performance': 0.30, 'estimated_potential': 0.25, 'estimated_regularity': 0.25, 'estimated_goals': 0.10, 'team_ranking': 0.10},
            'MID': {'estimated_performance': 0.25, 'estimated_potential': 0.25, 'estimated_regularity': 0.20, 'estimated_goals': 0.15, 'team_ranking': 0.15},
            'FWD': {'estimated_performance': 0.20, 'estimated_potential': 0.25, 'estimated_regularity': 0.15, 'estimated_goals': 0.25, 'team_ranking': 0.15}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.3},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.4},
            'MID': {'max_proportional_bonus_at_pvs100': 0.6},
            'FWD': {'max_proportional_bonus_at_pvs100': 0.8}
        }
    },
    "Potential Focus": {
        "kpi_weights": {
            'GK': {'estimated_performance': 0.20, 'estimated_potential': 0.60, 'estimated_regularity': 0.20, 'estimated_goals': 0.0, 'team_ranking': 0.0},
            'DEF': {'estimated_performance': 0.15, 'estimated_potential': 0.55, 'estimated_regularity': 0.15, 'estimated_goals': 0.05, 'team_ranking': 0.10},
            'MID': {'estimated_performance': 0.10, 'estimated_potential': 0.55, 'estimated_regularity': 0.15, 'estimated_goals': 0.10, 'team_ranking': 0.10},
            'FWD': {'estimated_performance': 0.05, 'estimated_potential': 0.50, 'estimated_regularity': 0.10, 'estimated_goals': 0.25, 'team_ranking': 0.10}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.25},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.35},
            'MID': {'max_proportional_bonus_at_pvs100': 0.5},
            'FWD': {'max_proportional_bonus_at_pvs100': 0.7}
        }
    },
    "Goal Focus": {
        "kpi_weights": {
            'GK': {'estimated_performance': 0.50, 'estimated_potential': 0.30, 'estimated_regularity': 0.20, 'estimated_goals': 0.0, 'team_ranking': 0.0},
            'DEF': {'estimated_performance': 0.20, 'estimated_potential': 0.10, 'estimated_regularity': 0.20, 'estimated_goals': 0.30, 'team_ranking': 0.20},
            'MID': {'estimated_performance': 0.15, 'estimated_potential': 0.10, 'estimated_regularity': 0.15, 'estimated_goals': 0.40, 'team_ranking': 0.20},
            'FWD': {'estimated_performance': 0.10, 'estimated_potential': 0.10, 'estimated_regularity': 0.10, 'estimated_goals': 0.60, 'team_ranking': 0.10}
        },
        "mrb_params_per_pos": {
            'GK': {'max_proportional_bonus_at_pvs100': 0.2},
            'DEF': {'max_proportional_bonus_at_pvs100': 0.3},
            'MID': {'max_proportional_bonus_at_pvs100': 0.4},
            'FWD': {'max_proportional_bonus_at_pvs100': 0.9}
        }
    }
}

# ----- Data helpers -----
def simplify_position(position: str) -> str:
    if pd.isna(position) or str(position).strip() == '':
        return 'UNKNOWN'
    pos = str(position).upper().strip()
    if pos == 'G': return 'GK'
    elif pos in ['D', 'DL', 'DC']: return 'DEF'
    elif pos in ['M', 'MD', 'MO']: return 'MID'
    elif pos == 'A': return 'FWD'
    else: return 'UNKNOWN'

def create_player_id(row) -> str:
    name = str(row.get('Joueur', '')).strip()
    simplified_pos = simplify_position(row.get('Poste', ''))
    club = str(row.get('Club', '')).strip()
    return f"{name}_{simplified_pos}_{club}"

def extract_rating_goals(rating_str):
    if pd.isna(rating_str) or str(rating_str).strip() == '' or str(rating_str).strip() == '0':
        return None, 0
    val_str = str(rating_str).strip()
    goals = val_str.count('*')
    clean_rating_str = re.sub(r'[()\*]', '', val_str)
    try:
        rating = float(clean_rating_str)
        return rating, goals
    except ValueError:
        return None, 0

def get_gameweek_columns(df_columns):
    gw_cols = [col for col in df_columns if re.fullmatch(r'D\d+', col)]
    gw_cols_sorted = sorted(gw_cols, key=lambda x: int(x[1:]))
    return gw_cols_sorted

def calculate_historical_kpis(df_hist: pd.DataFrame) -> pd.DataFrame:
    rdf = df_hist.copy()
    all_gws = get_gameweek_columns(df_hist.columns)
    rdf[['estimated_performance', 'estimated_potential', 'estimated_regularity', 'estimated_goals']] = 0.0
    for idx, row in rdf.iterrows():
        ratings, goals = [], 0
        games_played = 0
        for gw_col in all_gws:
            rating, game_goals = extract_rating_goals(row.get(gw_col))
            if rating is not None:
                ratings.append(rating)
                goals += game_goals
                games_played += 1
        if ratings:
            rdf.at[idx, 'estimated_performance'] = np.mean(ratings)
            rdf.at[idx, 'estimated_potential'] = np.mean(sorted(ratings, reverse=True)[:5]) if len(ratings) >= 5 else np.mean(ratings)
            rdf.at[idx, 'estimated_regularity'] = (games_played / len(all_gws) * 100) if all_gws else 0
            rdf.at[idx, 'estimated_goals'] = goals
    return rdf

def normalize_kpis(df_all, max_perf, max_pot, max_reg, max_goals):
    rdf = df_all.copy()
    rdf['norm_estimated_performance'] = np.clip(rdf['estimated_performance'] / max_perf * 100 if max_perf>0 else 0, 0, 100)
    rdf['norm_estimated_potential']   = np.clip(rdf['estimated_potential'] / max_pot * 100 if max_pot>0 else 0, 0, 100)
    rdf['norm_estimated_regularity']  = np.clip(rdf['estimated_regularity'] / max_reg * 100 if max_reg>0 else 0, 0, 100)
    rdf['norm_estimated_goals']       = np.clip(rdf['estimated_goals'] / max_goals * 100 if max_goals>0 else 0, 0, 100)
    rdf['norm_team_ranking']          = rdf['team_ranking']
    return rdf

def calculate_pvs(df, weights):
    rdf = df.copy()
    rdf['pvs'] = 0.0
    for pos in ['GK', 'DEF', 'MID', 'FWD']:
        mask = rdf['simplified_position'] == pos
        if not mask.any():
            continue
        w = weights[pos]
        total_weight = sum(w.values())
        if total_weight == 0:
            total_weight = 1.0
        pvs_raw = (
            rdf.loc[mask, 'norm_estimated_performance'] * w.get('estimated_performance', 0) +
            rdf.loc[mask, 'norm_estimated_potential'] * w.get('estimated_potential', 0) +
            rdf.loc[mask, 'norm_estimated_regularity'] * w.get('estimated_regularity', 0) +
            rdf.loc[mask, 'norm_estimated_goals'] * w.get('estimated_goals', 0) +
            rdf.loc[mask, 'norm_team_ranking'] * w.get('team_ranking', 0)
        )
        rdf.loc[mask, 'pvs'] = (pvs_raw / total_weight).clip(0, 100)
    return rdf

def calculate_mrb(df, mrb_params_per_pos):
    rdf = df.copy()
    rdf['mrb'] = rdf['Cote']
    for pos, params in mrb_params_per_pos.items():
        mask = rdf['simplified_position'] == pos
        if not mask.any(): continue
        max_prop_bonus = params.get('max_proportional_bonus_at_pvs100', 0.5)
        def _calc_mrb(row):
            cote = int(row['Cote']); pvs = float(row['pvs'])
            mrb_float = cote * (1 + (pvs/100)*max_prop_bonus)
            return int(round(min(mrb_float, cote*2)))
        rdf.loc[mask, 'mrb'] = rdf.loc[mask].apply(_calc_mrb, axis=1)
    rdf['mrb'] = rdf['mrb'].astype(int)
    safe_mrb = rdf['mrb'].replace(0, 1).astype(float)
    rdf['value_per_cost'] = rdf['pvs'] / safe_mrb
    rdf['value_per_cost'].fillna(0, inplace=True)
    return rdf

def save_dict_to_download_button(data_dict, label, fname):
    bio = io.BytesIO()
    bio.write(json.dumps(data_dict, indent=2).encode('utf-8'))
    bio.seek(0)
    st.download_button(label, data=bio, file_name=fname, mime='application/json')

def load_dict_from_file(uploaded_file):
    if uploaded_file is None:
        return {}
    try:
        content = uploaded_file.read()
        return json.loads(content.decode('utf-8'))
    except Exception as e:
        st.error(f"Could not load file: {e}")
        return {}

# ---- PLAYER PERFORMANCE VISUALIZATION ----
def plot_player_performance(player_row, df_hist):
    if not player_row.get('is_historical', False):
        st.info("No historical data available for new players")
        return
    hist_row = df_hist[df_hist['player_id'] == player_row['player_id']]
    if hist_row.empty:
        st.info("Historical data not found for this player")
        return
    hist_row = hist_row.iloc[0]
    gw_cols = get_gameweek_columns(hist_row.index)
    ratings, goals, gameweeks = [], [], []
    for gw in gw_cols:
        r, g = extract_rating_goals(hist_row[gw])
        if r is not None:
            ratings.append(r)
            goals.append(g)
            gameweeks.append(int(gw[1:]))
    if not ratings:
        st.info("No performance data available for this player")
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
    sns.lineplot(x=gameweeks, y=ratings, ax=ax1, marker='o')
    ax1.set_title(f"{player_row['Joueur']} Ratings per GW")
    ax1.set_ylabel("Rating"); ax1.set_ylim(0, 10); ax1.grid(True, linestyle='--', alpha=0.7)
    sns.barplot(x=gameweeks, y=goals, ax=ax2, color="#2563eb")
    ax2.set_title(f"{player_row['Joueur']} Goals per GW")
    ax2.set_xlabel("Gameweek"); ax2.set_ylabel("Goals"); ax2.grid(True, linestyle='--', alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)

# ---- MANUAL SQUAD LOGIC ----
def add_player_to_squad(player_id):
    if "manual_squad" not in st.session_state:
        st.session_state.manual_squad = []
    if player_id not in st.session_state.manual_squad:
        st.session_state.manual_squad.append(player_id)

def remove_player_from_squad(player_id):
    if "manual_squad" in st.session_state and player_id in st.session_state.manual_squad:
        st.session_state.manual_squad.remove(player_id)

# ---- MAIN APP ----
def main():
    st.markdown('<h1 class="main-header">🌟 MPG Manual Squad Builder</h1>', unsafe_allow_html=True)

    if 'manual_squad' not in st.session_state:
        st.session_state.manual_squad = []
    if 'selected_player' not in st.session_state:
        st.session_state.selected_player = None

    # ---- SIDEBAR: File Inputs and Settings ----
    with st.sidebar:
        st.markdown('<h2 class="section-header" style="margin-top:0;">⚙️ Data Files</h2>', unsafe_allow_html=True)
        hist_file = st.file_uploader("Last Season Player Data (CSV/Excel)", type=['csv','xlsx','xls'], key="hist_file")
        new_file = st.file_uploader("New Season Players File (CSV/Excel)", type=['csv','xlsx','xls'], key="new_file")
        st.markdown("---")
        st.markdown("#### 💾 Save/Load Squad")
        save_dict_to_download_button(st.session_state.manual_squad, "Download Squad", "squad.json")
        squad_upload = st.file_uploader("Load Squad", type=["json"], key="squad_upload")
        if squad_upload:
            loaded = load_dict_from_file(squad_upload)
            if isinstance(loaded, list):
                st.session_state.manual_squad = loaded
                st.success("Squad loaded!")
        st.markdown("---")
        st.caption("Upload player files and build your squad by clicking + and - in the lists.")

    # ---- LOAD DATA ----
    df_hist, df_new = None, None
    if hist_file:
        try:
            df_hist = pd.read_excel(hist_file) if hist_file.name.endswith(('xlsx','xls')) else pd.read_csv(hist_file)
        except Exception as e:
            st.error(f"Could not read historical file: {e}")
    if new_file:
        try:
            df_new = pd.read_excel(new_file) if new_file.name.endswith(('xlsx','xls')) else pd.read_csv(new_file)
        except Exception as e:
            st.error(f"Could not read new season file: {e}")
    if not (df_hist is not None and df_new is not None):
        st.info("Upload BOTH last season and new season player files to start.")
        return

    # ---- PREPARE DATA ----
    df_hist['simplified_position'] = df_hist['Poste'].apply(simplify_position)
    df_hist['player_id'] = df_hist.apply(create_player_id, axis=1)
    df_new['simplified_position'] = df_new['Poste'].apply(simplify_position)
    df_new['player_id'] = df_new.apply(create_player_id, axis=1)
    df_hist['Cote'] = pd.to_numeric(df_hist['Cote'], errors='coerce').fillna(1).clip(lower=1).round().astype(int)
    df_new['Cote'] = pd.to_numeric(df_new['Cote'], errors='coerce').fillna(1).clip(lower=1).round().astype(int)
    hist_pids = set(df_hist['player_id'])
    df_new['is_historical'] = df_new['player_id'].isin(hist_pids)
    df_hist_kpis = calculate_historical_kpis(df_hist)

    # New player scores
    if 'new_player_scores' not in st.session_state:
        st.session_state.new_player_scores = {}
    new_players = df_new[~df_new['is_historical']]
    for idx, row in new_players.iterrows():
        pid = row['player_id']
        if pid not in st.session_state.new_player_scores:
            st.session_state.new_player_scores[pid] = {
                "estimated_performance": 0,
                "estimated_potential": 0,
                "estimated_regularity": 0,
                "estimated_goals": 0,
            }

    # Club tiers
    all_clubs = sorted(df_new['Club'].unique())
    if 'club_tiers' not in st.session_state:
        st.session_state.club_tiers = {club: "Average" for club in all_clubs}
    else:
        for club in all_clubs:
            if club not in st.session_state.club_tiers:
                st.session_state.club_tiers[club] = "Average"

    # ---- Club & New player config in sidebar ----
    with st.sidebar.expander("🏅 Assign Club Tiers", expanded=False):
        for club in all_clubs:
            tier = st.selectbox(
                club, 
                CLUB_TIERS_LABELS, 
                index=CLUB_TIERS_LABELS.index(st.session_state.club_tiers.get(club,"Average")), 
                key=f"clubtier_{club}"
            )
            st.session_state.club_tiers[club] = tier

    with st.sidebar.expander("🆕 Assign Scores to New Players", expanded=False):
        if not new_players.empty:
            st.write("Rate new players (0, 25, 50, 75, 100% of max historical for each KPI):")
            for i, nprow in new_players.iterrows():
                pid = nprow['player_id']
                st.markdown(f"**{nprow['Joueur']} ({nprow['simplified_position']} - {nprow['Club']})**")
                for kpi, maxval, label in [
                    ("estimated_performance", df_hist_kpis['estimated_performance'].max(), "Performance"),
                    ("estimated_potential", df_hist_kpis['estimated_potential'].max(), "Potential"),
                    ("estimated_regularity", df_hist_kpis['estimated_regularity'].max(), "Regularity"),
                    ("estimated_goals", df_hist_kpis['estimated_goals'].max(), "Goals")
                ]:
                    sel = st.selectbox(
                        label, 
                        NEW_PLAYER_SCORE_OPTIONS,
                        index=NEW_PLAYER_SCORE_OPTIONS.index(st.session_state.new_player_scores[pid][kpi]),
                        key=f"{pid}_{kpi}"
                    )
                    st.session_state.new_player_scores[pid][kpi] = sel
        else:
            st.info("No new players to rate.")

    # ---- Merge all player data ----
    merged_rows = []
    for idx, row in df_new.iterrows():
        base = row.to_dict()
        club = base['Club']
        base['team_ranking'] = CLUB_TIERS[st.session_state.club_tiers[club]]
        if base['is_historical']:
            hist_row = df_hist_kpis[df_hist_kpis['player_id']==base['player_id']]
            for col in ['estimated_performance','estimated_potential','estimated_regularity','estimated_goals']:
                base[col] = float(hist_row.iloc[0][col]) if not hist_row.empty else 0.0
        else:
            pid = base['player_id']
            for kpi in ['estimated_performance','estimated_potential','estimated_regularity','estimated_goals']:
                score_pct = st.session_state.new_player_scores[pid][kpi]
                if kpi == "estimated_performance": maxval = df_hist_kpis['estimated_performance'].max()
                elif kpi == "estimated_potential": maxval = df_hist_kpis['estimated_potential'].max()
                elif kpi == "estimated_regularity": maxval = df_hist_kpis['estimated_regularity'].max()
                elif kpi == "estimated_goals": maxval = df_hist_kpis['estimated_goals'].max()
                base[kpi] = (score_pct/100) * maxval
        merged_rows.append(base)
    df_all = pd.DataFrame(merged_rows)

    # ---- Normalize and calculate metrics ----
    max_perf = df_hist_kpis['estimated_performance'].max()
    max_pot  = df_hist_kpis['estimated_potential'].max()
    max_reg  = df_hist_kpis['estimated_regularity'].max()
    max_goals= df_hist_kpis['estimated_goals'].max()
    profile = PREDEFINED_PROFILES["Balanced Value"]
    df_all = normalize_kpis(df_all, max_perf, max_pot, max_reg, max_goals)
    df_all = calculate_pvs(df_all, profile["kpi_weights"])
    df_all = calculate_mrb(df_all, profile["mrb_params_per_pos"])

    # ---- FILTER/SORT CONTROLS ----
    st.markdown('<h2 class="section-header">📋 Player Database</h2>', unsafe_allow_html=True)
    col1, col2, col3, col4 = st.columns(4)
    pos_filter = col1.multiselect("Position", options=df_all['simplified_position'].unique().tolist(), default=[])
    club_filter = col2.multiselect("Club", options=df_all['Club'].unique().tolist(), default=[])
    sort_col = col3.selectbox("Sort by", options=["pvs", "mrb", "estimated_performance", "estimated_potential", "estimated_regularity", "estimated_goals"], index=0)
    sort_asc = col4.checkbox("Ascending?", False)
    filtered = df_all.copy()
    if pos_filter:
        filtered = filtered[filtered['simplified_position'].isin(pos_filter)]
    if club_filter:
        filtered = filtered[filtered['Club'].isin(club_filter)]
    filtered = filtered.sort_values(by=sort_col, ascending=sort_asc)

    # ---- PLAYER TABLE WITH "+" BUTTON ----
    st.write("**Add players to your squad:**")
    for idx, row in filtered.iterrows():
        cols = st.columns([3,2,1,1,1])
        cols[0].write(f"**{row['Joueur']}** ({row['simplified_position']}, {row['Club']})")
        cols[1].write(f"PVS: {row['pvs']:.1f} | Bid: €{row['mrb']}")
        if row['player_id'] not in st.session_state.manual_squad:
            if cols[2].button("+", key=f"add_{row['player_id']}"):
                add_player_to_squad(row['player_id'])
        else:
            cols[2].write("✔️ In squad")
        if cols[3].button("Details", key=f"info_{row['player_id']}"):
            st.session_state.selected_player = row['player_id']

    # ---- SQUAD TABLE WITH "-" BUTTON ----
    st.markdown('<h2 class="section-header">🏆 Your Squad</h2>', unsafe_allow_html=True)
    squad_df = df_all[df_all['player_id'].isin(st.session_state.manual_squad)]
    total_cost = squad_df['mrb'].sum() if not squad_df.empty else 0
    st.write(f"**Total Cost:** €{total_cost}")
    for idx, row in squad_df.iterrows():
        cols = st.columns([3,2,1,1])
        cols[0].write(f"**{row['Joueur']}** ({row['simplified_position']}, {row['Club']})")
        cols[1].write(f"PVS: {row['pvs']:.1f} | Bid: €{row['mrb']}")
        if cols[2].button("-", key=f"remove_{row['player_id']}"):
            remove_player_from_squad(row['player_id'])
        if cols[3].button("Details Squad", key=f"info_sq_{row['player_id']}"):
            st.session_state.selected_player = row['player_id']

    # ---- PLAYER DETAIL PANEL ----
    if st.session_state.selected_player:
        selected_row = df_all[df_all['player_id'] == st.session_state.selected_player]
        if not selected_row.empty:
            st.sidebar.markdown(f"### Player Details: {selected_row.iloc[0]['Joueur']}")
            st.sidebar.write(selected_row.iloc[0][[
                'Joueur', 'Club', 'simplified_position', 'pvs', 'mrb',
                'estimated_performance','estimated_potential','estimated_regularity','estimated_goals'
            ]])
            plot_player_performance(selected_row.iloc[0], df_hist)

if __name__ == "__main__":
    main()
