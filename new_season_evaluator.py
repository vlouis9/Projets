import streamlit as st
import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple

# --- Configuration ---
st.set_page_config(page_title="MPG Auction Helper - New Season", layout="wide")
st.title("🚀 MPG Auction Helper (New Season)")

DEFAULT_BUDGET = 500
MINIMA = {'GK': 2, 'DEF': 6, 'MID': 6, 'FWD': 4}
FORMATIONS = {
    '4-4-2': {'GK': 1, 'DEF': 4, 'MID': 4, 'FWD': 2},
    '4-3-3': {'GK': 1, 'DEF': 4, 'MID': 3, 'FWD': 3},
    '3-5-2': {'GK': 1, 'DEF': 3, 'MID': 5, 'FWD': 2},
    '3-4-3': {'GK': 1, 'DEF': 3, 'MID': 4, 'FWD': 3},
    '4-5-1': {'GK': 1, 'DEF': 4, 'MID': 5, 'FWD': 1},
    '5-3-2': {'GK': 1, 'DEF': 5, 'MID': 3, 'FWD': 2},
    '5-4-1': {'GK': 1, 'DEF': 5, 'MID': 4, 'FWD': 1}
}
PROFILES = {
    'Balanced Value': {
        'n_recent_games': 5,
        'min_recent_played': 1,
        'kpi_weights': {
            'GK': {'recent_avg': 0.05, 'season_avg': 0.70, 'regularity': 0.25, 'recent_goals': 0.0, 'season_goals': 0.0},
            'DEF': {'recent_avg': 0.25, 'season_avg': 0.25, 'regularity': 0.25, 'recent_goals': 0.0, 'season_goals': 0.0},
            'MID': {'recent_avg': 0.20, 'season_avg': 0.20, 'regularity': 0.15, 'recent_goals': 0.15, 'season_goals': 0.15},
            'FWD': {'recent_avg': 0.15, 'season_avg': 0.15, 'regularity': 0.10, 'recent_goals': 0.25, 'season_goals': 0.25}
        },
        'mrb_params': {'GK': 0.3, 'DEF': 0.4, 'MID': 0.6, 'FWD': 1.0}
    },
    'Custom': None
}

# --- Utility Functions ---
def simplify_position(pos):
    if pd.isna(pos): return 'UNKNOWN'
    p = pos.strip().upper()
    if p == 'G': return 'GK'
    if p in ['D', 'DC', 'DL', 'DF']: return 'DEF'
    if p in ['M', 'MD', 'MO', 'MC']: return 'MID'
    if p in ['A', 'AT', 'FW']: return 'FWD'
    return 'UNKNOWN'

def create_player_id(r):
    return f"{r['Joueur'].strip()}_{simplify_position(r['Poste'])}_{r['Club'].strip()}"

def extract_rating(val):
    if pd.isna(val) or val in ['0', '']:
        return None, 0
    s = str(val).strip()
    goals = s.count('*')
    clean = re.sub(r"[()*]", "", s)
    try: return float(clean), goals
    except: return None, 0

# --- Load and process data ---
def load_data(f):
    df = pd.read_excel(f) if f.name.endswith(('xls', 'xlsx')) else pd.read_csv(f)
    df['simplified_position'] = df['Poste'].apply(simplify_position)
    df['player_id'] = df.apply(create_player_id, axis=1)
    df['Cote'] = pd.to_numeric(df['Cote'], errors='coerce').fillna(1).astype(int)
    df['Club'] = df['Club'].astype(str)
    return df

def compute_kpis(df, cols):
    result = []
    for _, r in df.iterrows():
        ratings, goals, played = [], 0, 0
        for c in cols:
            rt, g = extract_rating(r[c])
            if rt is not None:
                ratings.append(rt)
                goals += g
                played += 1
        avg = np.mean(ratings) if ratings else 0
        pot = np.mean(sorted(ratings, reverse=True)[:5]) if ratings else 0
        reg = played / len(cols) * 100 if cols else 0
        result.append({'player_id': r['player_id'], 'est_perf': avg * 10, 'est_potential': pot * 10, 'est_regularity': reg, 'est_goals': goals})
    return pd.DataFrame(result)

def evaluate_players(df, weights, mrb_bonus):
    df['pvs'] = df.apply(lambda r: sum(r[k]*weights[r['simplified_position']][k] for k in weights[r['simplified_position']]), axis=1)
    df['mrb'] = df.apply(lambda r: int(round(min(max(r['Cote'], r['Cote'] * (1 + mrb_bonus[r['simplified_position']] * r['pvs'] / 100)), 2 * r['Cote']))), axis=1)
    df['value_per_cost'] = df['pvs'] / df['mrb']
    return df

def select_squad(df, formation, minima, budget):
    squad, rem = [], df.copy()
    for pos, n in minima.items():
        sel = rem[rem['simplified_position'] == pos].nlargest(n, 'pvs')
        squad.append(sel)
        rem = rem.drop(sel.index)
    for pos, n in formation.items():
        have = pd.concat(squad)[lambda x: x['simplified_position'] == pos]
        needed = max(0, n - len(have))
        sel = rem[rem['simplified_position'] == pos].nlargest(needed, 'pvs')
        squad.append(sel)
        rem = rem.drop(sel.index)
    total = pd.concat(squad)
    while total['mrb'].sum() > budget:
        worst = total.nsmallest(1, 'value_per_cost')
        total = total.drop(worst.index)
        subs = df.drop(total.index)
        candidate = subs[subs['mrb'] <= budget - total['mrb'].sum()].nlargest(1, 'pvs')
        if not candidate.empty:
            total = pd.concat([total, candidate])
        else:
            break
    return total, {'total_cost': total['mrb'].sum(), 'remaining_budget': budget - total['mrb'].sum(), 'total_pvs': total['pvs'].sum()}

# --- Sidebar UI ---
st.sidebar.header("📁 Upload Data")
hist_file = st.sidebar.file_uploader("Last Season", type=["csv", "xls", "xlsx"])
new_file = st.sidebar.file_uploader("New Season", type=["csv", "xls", "xlsx"])
profile_key = st.sidebar.selectbox("Profile", list(PROFILES.keys()))
formation_key = st.sidebar.selectbox("Formation", list(FORMATIONS.keys()))
budget = st.sidebar.number_input("Budget", value=DEFAULT_BUDGET)

# --- Main Logic ---
if hist_file and new_file:
    df_hist = load_data(hist_file)
    df_new = load_data(new_file)
    df_hist_ids = set(df_hist['player_id'])
    df_new_ids = set(df_new['player_id'])
    known = df_new_ids & df_hist_ids
    new_only = df_new_ids - df_hist_ids

    # KPI Computation
    rating_cols = [c for c in df_hist.columns if re.fullmatch(r'D\\d+', c)]
    df_hist_kpis = compute_kpis(df_hist[df_hist['player_id'].isin(known)], rating_cols)
    max_vals = df_hist_kpis[['est_perf', 'est_potential', 'est_regularity', 'est_goals']].max()

    # Manual Sliders for new players
    st.subheader("✍️ Manual Input for New Players")
    manual = []
    for _, r in df_new[df_new['player_id'].isin(new_only)].iterrows():
        st.markdown(f"**{r['Joueur']} ({r['simplified_position']}, {r['Club']})**")
        perf = st.slider(f"Performance %", 0, 100, 50, 25, key=f"perf_{r['player_id']}")
        pot = st.slider(f"Potential %", 0, 100, 50, 25, key=f"pot_{r['player_id']}")
        reg = st.slider(f"Regularity %", 0, 100, 50, 25, key=f"reg_{r['player_id']}")
        goals = st.slider(f"Goals %", 0, 100, 0, 25, key=f"goals_{r['player_id']}")
        manual.append({**r[['player_id', 'Joueur', 'simplified_position', 'Club', 'Cote']].to_dict(),
                       'est_perf': perf / 100 * max_vals['est_perf'],
                       'est_potential': pot / 100 * max_vals['est_potential'],
                       'est_regularity': reg / 100 * max_vals['est_regularity'],
                       'est_goals': goals / 100 * max_vals['est_goals']})
    df_manual = pd.DataFrame(manual)

    # Merge known and new
    df_known = df_new[df_new['player_id'].isin(known)][['player_id', 'Joueur', 'simplified_position', 'Club', 'Cote']]
    df_known = df_known.merge(df_hist_kpis, on='player_id')
    df_eval = pd.concat([df_known, df_manual], ignore_index=True)

    # Club tier assignment
    st.sidebar.header("🏆 Club Tiers")
    clubs = sorted(df_new['Club'].unique())
    tier_map = {}
    tiers = {'Winner': 100, 'European': 75, 'Average': 50, 'Relegation': 25}
    for club in clubs:
        tier_map[club] = tiers[st.sidebar.selectbox(club, list(tiers.keys()), index=2)]
    df_eval['team_rank'] = df_eval['Club'].map(tier_map)

    # Profile weights
    if profile_key == 'Custom':
        st.subheader("🎛️ Custom Weights")
        prof = {'kpi_weights': {}, 'mrb_params': {}, 'n_recent_games': 5, 'min_recent_played': 1}
        for pos in ['GK', 'DEF', 'MID', 'FWD']:
            st.markdown(f"**{pos} KPI Weights**")
            prof['kpi_weights'][pos] = {
                'recent_avg': st.slider(f"{pos} Recent Avg", 0.0, 1.0, 0.2, 0.05),
                'season_avg': st.slider(f"{pos} Season Avg", 0.0, 1.0, 0.2, 0.05),
                'regularity': st.slider(f"{pos} Regularity", 0.0, 1.0, 0.2, 0.05),
                'recent_goals': st.slider(f"{pos} Recent Goals", 0.0, 1.0, 0.1, 0.05),
                'season_goals': st.slider(f"{pos} Season Goals", 0.0, 1.0, 0.1, 0.05)
            }
            prof['mrb_params'][pos] = st.slider(f"{pos} MRB Bonus", 0.0, 2.0, 1.0, 0.1)
    else:
        prof = PROFILES[profile_key]

    df_eval = evaluate_players(df_eval, prof['kpi_weights'], prof['mrb_params'])

    st.subheader("📊 Evaluated Players")
    st.dataframe(df_eval)

    squad, summary = select_squad(df_eval, FORMATIONS[formation_key], MINIMA, budget)

    st.subheader("🏆 Suggested Squad")
    st.dataframe(squad)

    st.subheader("📈 Summary")
    st.json(summary)
else:
    st.info("Upload both last season and new season files to proceed.")
