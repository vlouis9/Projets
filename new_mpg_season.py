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

# ---- CSS for compact UI ----
st.markdown("""
<style>
.main-header {
    font-size: 2.5rem; 
    font-weight: 800; 
    text-align: center; 
    margin-bottom: 1.5rem;
    padding-bottom: 0.5rem;
    border-bottom: 3px solid #10b981;
    background: linear-gradient(90deg, #2563eb, #8b5cf6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.section-header {
    font-size: 1.5rem;
    font-weight: 700;
    color: #2563eb;
    margin-top: 1.5rem;
    margin-bottom: 1rem;
    padding-left: 0.5rem;
    border-left: 4px solid #10b981;
}
.squad-summary-card {
    background: #f8fafc;
    border-radius: 8px;
    margin-bottom: 8px;
    padding: 0.7em 1em 0.7em 1em;
    font-size: 1.05em;
    border-left: 5px solid #2563eb;
}
.position-badge {
    border-radius: 12px;
    padding: 0.07em 0.6em;
    font-size: 0.98em;
    font-weight: 700;
    color: #fff;
    display: inline-block;
}
.GK-badge { background: #3b82f6; }
.DEF-badge { background: #22c55e; }
.MID-badge { background: #f59e42; }
.FWD-badge { background: #f43f5e; }
.player-details-box {
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    background: #f8fafc;
    margin: 0.5em 0 1em 0;
    padding: 1em;
}
.compact-btn {
    font-size: 1.0em !important;
    min-width: 0.8em !important;
    padding: 0.1em 0.4em !important;
    margin: 0 0.2em 0 0 !important;
}
.stDataFrame td { font-size: 0.98em; }
</style>
""", unsafe_allow_html=True)

# ---- CONSTANTS & PROFILES ----
CLUB_TIERS = {
    "Winner": 100, "European": 75, "Average": 50, "Relegation": 25
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

# ---- HELPERS ----
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
    return sorted(gw_cols, key=lambda x: int(x[1:]))

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
    rdf['norm_estimated_performance'] = np.clip(rdf['estimated_performance'] / max_perf * 100 if max_perf else 0, 0, 100)
    rdf['norm_estimated_potential'] = np.clip(rdf['estimated_potential'] / max_pot * 100 if max_pot else 0, 0, 100)
    rdf['norm_estimated_regularity'] = np.clip(rdf['estimated_regularity'] / max_reg * 100 if max_reg else 0, 0, 100)
    rdf['norm_estimated_goals'] = np.clip(rdf['estimated_goals'] / max_goals * 100 if max_goals else 0, 0, 100)
    rdf['norm_team_ranking'] = rdf['team_ranking']
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

def add_player_to_squad(player_id):
    if "manual_squad" not in st.session_state:
        st.session_state.manual_squad = []
    if player_id not in st.session_state.manual_squad:
        st.session_state.manual_squad.append(player_id)

def remove_player_from_squad(player_id):
    if "manual_squad" in st.session_state and player_id in st.session_state.manual_squad:
        st.session_state.manual_squad.remove(player_id)

def squad_summary(df):
    if df.empty:
        return {}
    pos_counts = df['simplified_position'].value_counts().to_dict()
    goals = df['estimated_goals'].sum()
    regularity = df['estimated_regularity'].mean() if not df.empty else 0
    return {
        "Total players": len(df),
        "Total cost (€)": df['mrb'].sum(),
        "Total PVS": round(df['pvs'].sum(),1),
        "Total goals": int(goals),
        "Avg performance": round(df['estimated_performance'].mean(),2),
        "Avg potential": round(df['estimated_potential'].mean(),2),
        "Avg regularity (%)": round(regularity,1),
        "Pos counts": pos_counts
    }

# ---- MAIN APP ----
def main():
    st.markdown('<h1 class="main-header">🌟 MPG Manual Squad Builder</h1>', unsafe_allow_html=True)

    # ---- Session State Defaults ----
    for key, default in {
        'manual_squad': [],
        'selected_player': None,
        'kpi_weights': PREDEFINED_PROFILES["Balanced Value"]["kpi_weights"],
        'mrb_params': PREDEFINED_PROFILES["Balanced Value"]["mrb_params_per_pos"],
        'profile_name': "Balanced Value"
    }.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # ---- Sidebar: Inputs & Customization ----
    with st.sidebar:
        st.markdown('<h2 class="section-header">⚙️ Data Files</h2>', unsafe_allow_html=True)
        hist_file = st.file_uploader("Last Season Player Data (CSV/Excel)", type=['csv','xlsx','xls'])
        new_file = st.file_uploader("New Season Players File (CSV/Excel)", type=['csv','xlsx','xls'])
        st.markdown("---")
        st.markdown("### 🎨 Settings Profile")
        profile_names = list(PREDEFINED_PROFILES.keys())
        selected_profile = st.selectbox("Select Profile", options=profile_names, index=profile_names.index(st.session_state['profile_name']))
        if selected_profile != st.session_state['profile_name']:
            st.session_state['profile_name'] = selected_profile
            st.session_state['kpi_weights'] = PREDEFINED_PROFILES[selected_profile]["kpi_weights"]
            st.session_state['mrb_params'] = PREDEFINED_PROFILES[selected_profile]["mrb_params_per_pos"]
        with st.expander("📊 KPI Weights (Customize)", expanded=(selected_profile=="Custom")):
            weights_ui = {}
            for pos in ['GK', 'DEF', 'MID', 'FWD']:
                st.markdown(f"<h6>{pos}</h6>", unsafe_allow_html=True)
                current_w = st.session_state['kpi_weights'][pos]
                weights_ui[pos] = {
                    kpi: st.slider(f"{kpi.replace('estimated_', '').capitalize()} ({pos})", 0.0, 1.0, float(current_w.get(kpi, 0)), 0.01)
                    for kpi in ['estimated_performance', 'estimated_potential', 'estimated_regularity', 'estimated_goals', 'team_ranking']
                }
            if selected_profile=="Custom":
                st.session_state['kpi_weights'] = weights_ui
        with st.expander("💰 MRB Parameters", expanded=(selected_profile=="Custom")):
            mrb_ui = {}
            for pos in ['GK', 'DEF', 'MID', 'FWD']:
                mrb = st.slider(f"Max Bonus at PVS 100 ({pos})", 0.0, 1.0, float(st.session_state['mrb_params'][pos]['max_proportional_bonus_at_pvs100']), 0.01)
                mrb_ui[pos] = {'max_proportional_bonus_at_pvs100': mrb}
            if selected_profile=="Custom":
                st.session_state['mrb_params'] = mrb_ui
        st.markdown("---")
        save_dict_to_download_button(st.session_state.manual_squad, "💾 Download Squad", "squad.json")
        squad_upload = st.file_uploader("⬆️ Load Squad", type=["json"], key="squad_upload")
        if squad_upload:
            loaded = load_dict_from_file(squad_upload)
            if isinstance(loaded, list):
                st.session_state.manual_squad = loaded
                st.success("Squad loaded!")
        st.markdown("---")
        st.caption("Upload player files and build your squad by clicking + and - in the lists.")

    if not hist_file or not new_file:
        st.info("Upload BOTH last season and new season player files to start.")
        return

    # ---- Data Load & Preparation ----
    df_hist = pd.read_excel(hist_file) if hist_file.name.endswith(('xlsx','xls')) else pd.read_csv(hist_file)
    df_new = pd.read_excel(new_file) if new_file.name.endswith(('xlsx','xls')) else pd.read_csv(new_file)
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

    # Merge all player data
    merged_rows = []
    filtered = df_all.copy()
    if pos_filter:
        filtered = filtered[filtered['simplified_position'].isin(pos_filter)]
    if club_filter:
        filtered = filtered[filtered['Club'].isin(club_filter)]
    for idx, row in filtered.iterrows():
        pid = row['player_id']
        # Toggle visible state
        if f"show_stats_{pid}" not in st.session_state:
            st.session_state[f"show_stats_{pid}"] = False
    
        col1, col2 = st.columns([10, 2])
        with col1:
            st.markdown(
                f"<div style='border:1px solid #e2e8f0; padding:8px; border-radius:6px; margin-bottom:6px;'>"
                f"<b>{row['Joueur']}</b> (<span class='position-badge {row['simplified_position']}-badge'>{row['simplified_position']}</span> - {row['Club']}) "
                f"| PVS: <b>{row['pvs']:.1f}</b> | MRB: €{row['mrb']}<br>"
                f"Rating: {row['estimated_performance']:.2f} | Potentiel: {row['estimated_potential']:.2f} | "
                f"Regularité: {row['estimated_regularity']:.1f}% | Buts: {row['estimated_goals']} "
                f"</div>", unsafe_allow_html=True
            )
    
        with col2:
            if row['player_id'] not in st.session_state.manual_squad:
                if st.button("➕", key=f"add_{pid}"):
                    add_player_to_squad(pid)
            else:
                st.markdown("✅ Ajouté")
    
            if st.button("🔍", key=f"toggle_stats_{pid}"):
                st.session_state[f"show_stats_{pid}"] = not st.session_state[f"show_stats_{pid}"]
    
        if st.session_state[f"show_stats_{pid}"]:
            with st.container():
                st.markdown(f"<div class='player-details-box'><b>Détails pour {row['Joueur']} ({row['Club']})</b><br>"
                            f"PVS: {row['pvs']:.2f} | MRB: €{row['mrb']}<br>"
                            f"Performance: {row['estimated_performance']:.2f} | Potentiel: {row['estimated_potential']:.2f} | "
                            f"Regularité: {row['estimated_regularity']:.1f}% | Buts: {row['estimated_goals']}<br></div>", unsafe_allow_html=True)
                plot_player_performance(row, df_hist)

    # Normalize and calculate metrics
    max_perf = df_hist_kpis['estimated_performance'].max()
    max_pot  = df_hist_kpis['estimated_potential'].max()
    max_reg  = df_hist_kpis['estimated_regularity'].max()
    max_goals= df_hist_kpis['estimated_goals'].max()
    df_all = normalize_kpis(df_all, max_perf, max_pot, max_reg, max_goals)
    df_all = calculate_pvs(df_all, st.session_state['kpi_weights'])
    df_all = calculate_mrb(df_all, st.session_state['mrb_params'])

    # ---- UI Layout: Squad & Database ----
    squad_col, db_col = st.columns([1,2])
    # --- Left: Squad Panel ---
    with squad_col:
        st.markdown('<h2 class="section-header">🏆 Your Squad</h2>', unsafe_allow_html=True)
        squad_df = df_all[df_all['player_id'].isin(st.session_state.manual_squad)]
        summary = squad_summary(squad_df)
        if summary:
            st.markdown(f"<div class='squad-summary-card'><b>Total:</b> {summary['Total players']} &nbsp; "
                        f"<b>Cost:</b> €{summary['Total cost (€)']} &nbsp; "
                        f"<b>PVS:</b> {summary['Total PVS']} &nbsp; "
                        f"<b>Goals:</b> {summary['Total goals']}<br>"
                        f"<b>Avg Perf:</b> {summary['Avg performance']} &nbsp; "
                        f"<b>Avg Pot:</b> {summary['Avg potential']} &nbsp; "
                        f"<b>Avg Reg:</b> {summary['Avg regularity (%)']}%<br>"
                        + " ".join([f"<span class='position-badge {p}-badge'>{p}: {c}</span>" for p,c in summary['Pos counts'].items()])
                        + "</div>", unsafe_allow_html=True)
        for idx, row in squad_df.iterrows():
            cols = st.columns([2,1,1,0.7,0.7])
            cols[0].markdown(f"{row['Joueur']} (<span class='position-badge {row['simplified_position']}-badge'>{row['simplified_position']}</span>, {row['Club']})", unsafe_allow_html=True)
            cols[1].markdown(f"PVS: <b>{row['pvs']:.1f}</b>", unsafe_allow_html=True)
            cols[2].markdown(f"€{row['mrb']}")
            if cols[3].button("➖", key=f"remove_{row['player_id']}"):
                remove_player_from_squad(row['player_id'])
            if cols[4].button("🔍", key=f"details_squad_{row['player_id']}"):
                st.session_state.selected_player = row['player_id']
            if st.session_state.selected_player == row['player_id']:
                with st.container():
                    st.markdown(f"<div class='player-details-box'><b>Details for {row['Joueur']} ({row['Club']})</b><br>"
                                f"PVS: {row['pvs']:.2f} | MRB: €{row['mrb']}<br>"
                                f"Performance: {row['estimated_performance']:.2f} | Potential: {row['estimated_potential']:.2f} | "
                                f"Regularity: {row['estimated_regularity']:.1f}% | Goals: {row['estimated_goals']}<br></div>", unsafe_allow_html=True)
                    plot_player_performance(row, df_hist)

    # --- Right: Player Database ---
    with db_col:
        st.markdown('<h2 class="section-header">📋 Player Database</h2>', unsafe_allow_html=True)
        pos_filter = st.multiselect("Position", options=df_all['simplified_position'].unique().tolist(), default=[])
        club_filter = st.multiselect("Club", options=df_all['Club'].unique().tolist(), default=[])
        sort_col = st.selectbox("Sort by", options=["pvs", "mrb", "estimated_performance", "estimated_potential", "estimated_regularity", "estimated_goals"], index=0)
        sort_asc = st.checkbox("Ascending?", False)
        filtered = df_all.copy()
        if pos_filter:
            filtered = filtered[filtered['simplified_position'].isin(pos_filter)]
        if club_filter:
            filtered = filtered[filtered['Club'].isin(club_filter)]
        filtered = filtered.sort_values(by=sort_col, ascending=sort_asc)
        # Table headers
        table_cols = ["Player", "Pos", "Club", "PVS", "MRB", "Avg Rating", "Regularity (%)", "Goals", "", ""]
        st.markdown("| " + " | ".join(table_cols) + " |")
        st.markdown("|" + " --- |"*len(table_cols))
        # Each row
        for idx, row in filtered.iterrows():
            row_vals = [
                row['Joueur'],
                f"<span class='position-badge {row['simplified_position']}-badge'>{row['simplified_position']}</span>",
                row['Club'],
                f"{row['pvs']:.1f}",
                f"€{row['mrb']}",
                f"{row['estimated_performance']:.2f}",
                f"{row['estimated_regularity']:.1f}",
                f"{row['estimated_goals']:.0f}",
            ]
            col_btns = st.columns(len(row_vals)+2)
            for i, val in enumerate(row_vals):
                col_btns[i].markdown(val, unsafe_allow_html=True)
            if row['player_id'] not in st.session_state.manual_squad:
                if col_btns[-2].button("➕", key=f"add_{row['player_id']}"):
                    add_player_to_squad(row['player_id'])
            else:
                col_btns[-2].write("✅")
            if col_btns[-1].button("🔍", key=f"details_{row['player_id']}"):
                st.session_state.selected_player = row['player_id']
            if st.session_state.selected_player == row['player_id']:
                with st.container():
                    st.markdown(f"<div class='player-details-box'><b>Details for {row['Joueur']} ({row['Club']})</b><br>"
                                f"PVS: {row['pvs']:.2f} | MRB: €{row['mrb']}<br>"
                                f"Performance: {row['estimated_performance']:.2f} | Potential: {row['estimated_potential']:.2f} | "
                                f"Regularity: {row['estimated_regularity']:.1f}% | Goals: {row['estimated_goals']}<br></div>", unsafe_allow_html=True)
                    plot_player_performance(row, df_hist)

if __name__ == "__main__":
    main()
