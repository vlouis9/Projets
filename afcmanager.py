import streamlit as st
import json
import pandas as pd
import plotly.graph_objects as go
import datetime

st.set_page_config(page_title="⚽ Gestion d'Équipe AFC", layout="wide")

# ---------- DATA HANDLING ----------
def load_data():
    if "afcdata" not in st.session_state:
        try:
            with open("afcdata.json", "r") as f:
                st.session_state.afcdata = json.load(f)
        except:
            st.session_state.afcdata = {"players": [], "lineups": [], "matches": []}

def save_data():
    with open("afcdata.json", "w") as f:
        json.dump(st.session_state.afcdata, f, indent=2)

load_data()

def export_data():
    return json.dumps(st.session_state.afcdata, indent=2)

def import_data(uploaded_file):
    try:
        st.session_state.afcdata = json.load(uploaded_file)
        save_data()
        st.success("Import réussi.")
    except Exception as e:
        st.error(f"Erreur d'import: {e}")

# ---------- FORMATIONS ----------
formations = {
    "4-2-3-1": [(0.5, 0.1), (0.3, 0.25), (0.7, 0.25), (0.2, 0.4), (0.4, 0.4), (0.6, 0.4), (0.8, 0.4), (0.3, 0.6), (0.5, 0.6), (0.7, 0.6), (0.5, 0.8)],
    "4-4-2": [(0.5, 0.1), (0.3, 0.25), (0.7, 0.25), (0.2, 0.4), (0.4, 0.4), (0.6, 0.4), (0.8, 0.4), (0.3, 0.6), (0.7, 0.6), (0.4, 0.8), (0.6, 0.8)],
    "4-3-3": [(0.5, 0.1), (0.3, 0.25), (0.7, 0.25), (0.2, 0.4), (0.4, 0.4), (0.6, 0.4), (0.8, 0.4), (0.25, 0.6), (0.5, 0.6), (0.75, 0.6), (0.5, 0.8)],
    "3-5-2": [(0.5, 0.1), (0.3, 0.3), (0.7, 0.3), (0.2, 0.45), (0.4, 0.45), (0.6, 0.45), (0.8, 0.45), (0.3, 0.6), (0.7, 0.6), (0.4, 0.8), (0.6, 0.8)],
    "3-4-3": [(0.5, 0.1), (0.3, 0.3), (0.7, 0.3), (0.2, 0.45), (0.4, 0.45), (0.6, 0.45), (0.8, 0.45), (0.25, 0.6), (0.5, 0.6), (0.75, 0.6), (0.5, 0.8)],
    "5-3-2": [(0.5, 0.1), (0.2, 0.3), (0.35, 0.3), (0.5, 0.3), (0.65, 0.3), (0.8, 0.3), (0.3, 0.5), (0.5, 0.5), (0.7, 0.5), (0.4, 0.8), (0.6, 0.8)],
}

# ---------- UTILITY FUNCTIONS ----------
def draw_pitch(players, formation, captain, title):
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=1, y1=1, fillcolor="green", line=dict(color="white"))
    for i, (x, y) in enumerate(formations[formation]):
        if i >= len(players):
            continue
        player = players[i]
        if player:
            color = "darkblue"
            text = f"{player['number']} {player['name']}"
            if player['name'] == captain:
                text += " ©️"
            fig.add_trace(go.Scatter(x=[x], y=[y], mode="markers+text", marker=dict(size=30, color=color), text=[text], textposition="top center", textfont=dict(color="white")))
    fig.update_layout(height=600, showlegend=False, margin=dict(l=20, r=20, t=20, b=20))
    st.plotly_chart(fig)

# ---------- STATS COMPUTATION ----------
def compute_player_stats():
    stats = {}
    for p in st.session_state.afcdata["players"]:
        stats[p["name"]] = {"Sélections": 0, "Homme du match": 0, "Note moyenne": 0, "Total notes": 0}

    for m in st.session_state.afcdata["matches"]:
        if not m["finished"]:
            continue
        notes = m["stats"].get("notes", {})
        best = m["stats"].get("homme_du_match", "")
        for p in m["players"]:
            if p and p["name"] in stats:
                stats[p["name"]]["Sélections"] += 1
                if p["name"] == best:
                    stats[p["name"]]["Homme du match"] += 1
                note = notes.get(p["name"], 0)
                stats[p["name"]]["Note moyenne"] += note
                stats[p["name"]]["Total notes"] += 1

    for s in stats.values():
        if s["Total notes"]:
            s["Note moyenne"] = round(s["Note moyenne"] / s["Total notes"], 2)
        else:
            s["Note moyenne"] = None
    return stats

# ---------- MAIN TABS ----------
tabs = st.tabs(["Base joueurs", "Compositions", "Matchs", "Statistiques"])

# ---- PLAYERS DATABASE ----
with tabs[0]:
    st.header("Base joueurs")
    df = pd.DataFrame(st.session_state.afcdata["players"])
    edited_df = st.data_editor(df, num_rows="dynamic", key="players_edit")
    if st.button("💾 Sauvegarder les modifications"):
        st.session_state.afcdata["players"] = edited_df.dropna(subset=["name"]).to_dict(orient="records")
        save_data()
        st.success("Joueurs sauvegardés.")

# ---- LINEUPS ----
with tabs[1]:
    subtab = st.radio("", ["Créer une composition", "Mes compositions"], horizontal=True)

    if subtab == "Créer une composition":
        name = st.text_input("Nom de la composition")
        formation = st.selectbox("Formation", list(formations.keys()))
        players = st.session_state.afcdata["players"]
        selected = []
        for i, pos in enumerate(formations[formation]):
            options = [""] + [p["name"] for p in players if p["name"] not in [s.get("name") for s in selected if s]]
            sel = st.selectbox(f"Poste {i+1}", options)
            number = st.number_input(f"Numéro pour {sel}", min_value=1, max_value=99, value=10) if sel else 0
            selected.append({"name": sel, "number": number} if sel else None)
        captain = st.selectbox("Capitaine", [p["name"] for p in selected if p])
        draw_pitch(selected, formation, captain, "Composition")
        if st.button("💾 Sauvegarder la composition") and name:
            st.session_state.afcdata["lineups"].append({"name": name, "formation": formation, "players": selected, "captain": captain})
            save_data()
            st.success("Composition sauvegardée.")

    else:
        for idx, compo in enumerate(st.session_state.afcdata["lineups"]):
            with st.expander(compo["name"]):
                draw_pitch(compo["players"], compo["formation"], compo["captain"], compo["name"])
                col1, col2 = st.columns(2)
                if col1.button(f"✏️ Modifier {compo['name']}", key=f"edit_{idx}"):
                    st.warning("Modification non encore implémentée.")
                if col2.button(f"🗑️ Supprimer {compo['name']}", key=f"del_{idx}"):
                    st.session_state.afcdata["lineups"].pop(idx)
                    save_data()
                    st.experimental_rerun()

# ---- MATCHES ----
with tabs[2]:
    subtab = st.radio("", ["Créer un match", "Mes matchs"], horizontal=True)

    if subtab == "Créer un match":
        name = st.text_input("Nom du match", value=f"Match {datetime.date.today()}")
        match_type = st.selectbox("Type", ["Championnat", "Coupe", "Amical"])
        opponent = st.text_input("Adversaire")
        date = st.date_input("Date", datetime.date.today())
        location = st.text_input("Lieu")

        use_lineup = st.selectbox("Charger une composition existante ?", ["Aucune"] + [l["name"] for l in st.session_state.afcdata["lineups"]])
        if use_lineup != "Aucune":
            lineup = next(l for l in st.session_state.afcdata["lineups"] if l["name"] == use_lineup)
            formation = lineup["formation"]
            selected = lineup["players"]
            captain = lineup["captain"]
        else:
            formation = st.selectbox("Formation", list(formations.keys()))
            players = st.session_state.afcdata["players"]
            selected = []
            for i, pos in enumerate(formations[formation]):
                options = [""] + [p["name"] for p in players if p["name"] not in [s.get("name") for s in selected if s]]
                sel = st.selectbox(f"Poste {i+1}", options, key=f"match_{i}")
                number = st.number_input(f"Numéro pour {sel}", min_value=1, max_value=99, value=10, key=f"num_{i}") if sel else 0
                selected.append({"name": sel, "number": number} if sel else None)
            captain = st.selectbox("Capitaine", [p["name"] for p in selected if p])

        draw_pitch(selected, formation, captain, "Composition du match")

        if st.button("💾 Sauvegarder le match") and name and opponent:
            st.session_state.afcdata["matches"].append({
                "name": name,
                "type": match_type,
                "opponent": opponent,
                "date": str(date),
                "location": location,
                "formation": formation,
                "players": selected,
                "captain": captain,
                "finished": False,
                "stats": {}
            })
            save_data()
            st.success("Match sauvegardé.")

    else:
        for idx, match in enumerate(st.session_state.afcdata["matches"]):
            with st.expander(f"{match['name']} - {match['opponent']} ({match['date']})"):
                draw_pitch(match["players"], match["formation"], match["captain"], match["name"])
                if match["finished"]:
                    st.info("Match terminé. Statistiques:")
                    st.write(match["stats"])
                else:
                    score = st.text_input("Score final", key=f"score_{idx}")
                    best = st.selectbox("Homme du match", [p["name"] for p in match["players"] if p])
                    notes = {}
                    for p in match["players"]:
                        if p:
                            notes[p["name"]] = st.slider(f"Note pour {p['name']}", 0, 10, 6)
                    if st.button("✅ Clôturer le match", key=f"end_{idx}"):
                        match["finished"] = True
                        match["stats"] = {"score": score, "homme_du_match": best, "notes": notes}
                        save_data()
                        st.success("Match clôturé.")

                if st.button("🗑️ Supprimer ce match", key=f"delm_{idx}"):
                    st.session_state.afcdata["matches"].pop(idx)
                    save_data()
                    st.experimental_rerun()

# ---- STATISTICS ----
with tabs[3]:
    st.header("Statistiques joueurs cumulées")
    stats = compute_player_stats()
    stats_df = pd.DataFrame.from_dict(stats, orient="index").reset_index().rename(columns={"index": "Nom"})
    st.dataframe(stats_df)
