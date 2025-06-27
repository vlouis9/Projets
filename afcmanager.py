import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

DATA_FILE = "afcdata.json"
PLAYER_COLS = ["Nom", "Poste", "Infos", "Numero"]
PLAYER_DEFAULTS = {"Nom": "", "Poste": "G", "Infos": "", "Numero": 10}

FORMATION = {
    "4-4-2": {"G": 1, "D": 4, "M": 4, "A": 2},
    "4-3-3": {"G": 1, "D": 4, "M": 3, "A": 3},
    "3-5-2": {"G": 1, "D": 3, "M": 5, "A": 2},
    "3-4-3": {"G": 1, "D": 3, "M": 4, "A": 3},
    "5-3-2": {"G": 1, "D": 5, "M": 3, "A": 2},
}

POSTES_ORDER = ["A", "M", "D", "G"]
DEFAULT_FORMATION = "4-4-2"
MAX_REMPLACANTS = 5

POSITION_COORDS = {
    "A": [(40, 10), (60, 10), (50, 5)],
    "M": [(25, 35), (40, 35), (60, 35), (75, 35), (50, 40)],
    "D": [(20, 65), (35, 65), (65, 65), (80, 65), (50, 70)],
    "G": [(50, 90)]
}

st.set_page_config(page_title="Gestion Équipe AFC", layout="centered")

def save_all():
    data = {
        "players": st.session_state.players.to_dict(orient="records"),
        "lineups": st.session_state.lineups,
        "matches": st.session_state.matches,
    }
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def reload_all():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        st.session_state.players = pd.DataFrame(data.get("players", []))
        st.session_state.lineups = data.get("lineups", {})
        st.session_state.matches = data.get("matches", {})
    else:
        st.session_state.players = pd.DataFrame(columns=PLAYER_COLS)
        st.session_state.lineups = {}
        st.session_state.matches = {}

if "players" not in st.session_state:
    reload_all()
if "lineups" not in st.session_state:
    reload_all()
if "matches" not in st.session_state:
    reload_all()

def terrain_visuel(formation, terrain):
    st.markdown("""<div style='position:relative;width:100%;max-width:600px;margin:auto;'>""", unsafe_allow_html=True)
    st.image("https://i.imgur.com/7nK6b1v.png", use_column_width=True)
    for poste, nb in FORMATION[formation].items():
        coords_list = POSITION_COORDS.get(poste, [])
        for i in range(nb):
            if i < len(coords_list):
                x, y = coords_list[i]
                joueur = terrain.get(poste, [None]*nb)[i]
                label = joueur if joueur else f"{poste}{i+1}"
                couleur = "#2ecc71" if joueur else "#ecf0f1"
                st.markdown(f"""
                <div style='position:absolute;left:{x}%;top:{y}%;transform:translate(-50%,-50%);
                background:{couleur};padding:6px 10px;border-radius:12px;border:1px solid #333;'>
                {label}
                </div>
                """, unsafe_allow_html=True)
    st.markdown("""</div>""", unsafe_allow_html=True)

def compute_player_stats(joueur_nom):
    buts = passes = cj = cr = selections = titularisations = note_sum = note_count = hdm = 0
    for match in st.session_state.matches.values():
        details = match.get("details", {})
        joueurs = [j for p in details for j in details.get(p, []) if isinstance(j, str) and j == joueur_nom]
        if joueur_nom in match.get("remplacants", []):
            selections += 1
        if joueurs:
            titularisations += 1
            selections += 1
        events = match.get("events", {})
        buts += events.get("buteurs", {}).get(joueur_nom, 0)
        passes += events.get("passeurs", {}).get(joueur_nom, 0)
        cj += events.get("cartons_jaunes", {}).get(joueur_nom, 0)
        cr += events.get("cartons_rouges", {}).get(joueur_nom, 0)
        if "notes" in events and joueur_nom in events["notes"]:
            note_sum += events["notes"][joueur_nom]
            note_count += 1
        if "homme_du_match" in match and match["homme_du_match"] == joueur_nom:
            hdm += 1
    note = round(note_sum / note_count, 2) if note_count else 0
    buts_passes = buts + passes
    decisif_par_match = round(buts_passes / selections, 2) if selections > 0 else 0
    return {
        "Buts": buts,
        "Passes": passes,
        "Décisif/match": decisif_par_match,
        "CJ": cj,
        "CR": cr,
        "Sélections": selections,
        "Titularisations": titularisations,
        "Note": note,
        "Hdm": hdm
    }

def remplaçants_interactif(key, titulaires):
    if f"remp_{key}" not in st.session_state:
        st.session_state[f"remp_{key}"] = [None] * MAX_REMPLACANTS
    remps = st.session_state[f"remp_{key}"]
    dispo = [n for n in st.session_state.players["Nom"] if n not in titulaires and n not in remps]
    for i in range(MAX_REMPLACANTS):
        current = remps[i]
        options = dispo + ([current] if current and current not in dispo else [])
        choix = st.selectbox(f"Remplaçant {i+1}", [""] + options, index=(options.index(current)+1) if current in options else 0, key=f"remp_choice_{key}_{i}")
        remps[i] = choix if choix else None
        dispo = [n for n in dispo if n != choix]
    st.session_state[f"remp_{key}"] = remps
    return [r for r in remps if r]

st.sidebar.title("⚽ Gestion Équipe AFC")
menu = st.sidebar.radio("Menu", ["Base Joueurs", "Compositions", "Matchs", "Export JSON"])

if menu == "Base Joueurs":
    st.title("Base de données joueurs")
    edited_df = st.data_editor(st.session_state.players, num_rows="dynamic", use_container_width=True)
    if st.button("Sauvegarder"):
        edited_df = edited_df.fillna("")
        edited_df = edited_df[edited_df["Nom"].str.strip() != ""]
        st.session_state.players = edited_df[PLAYER_COLS]
        save_all()
        st.success("Base sauvegardée")

    st.subheader("Statistiques joueurs")
    stats = []
    for _, row in st.session_state.players.iterrows():
        s = compute_player_stats(row["Nom"])
        stats.append({**row, **s})
    st.dataframe(pd.DataFrame(stats))

elif menu == "Compositions":
    st.title("Créer une composition")
    nom = st.text_input("Nom de la compo")
    formation = st.selectbox("Formation", list(FORMATION.keys()))
    terrain = {poste: [None]*nb for poste, nb in FORMATION[formation].items()}
    joueurs_dispo = st.session_state.players["Nom"].tolist()
    for poste, nb in FORMATION[formation].items():
        for i in range(nb):
            choix = st.selectbox(f"{poste}{i+1}", [""] + joueurs_dispo, key=f"{poste}_{i}")
            if choix:
                terrain[poste][i] = choix
    st.subheader("Aperçu terrain")
    terrain_visuel(formation, terrain)
    if st.button("Sauvegarder la compo"):
        if not nom.strip():
            st.warning("Donnez un nom à la compo")
        else:
            st.session_state.lineups[nom] = {"formation": formation, "details": terrain}
            save_all()
            st.success("Composition sauvegardée")
    st.subheader("Compositions existantes")
    if st.session_state.lineups:
        for k, v in st.session_state.lineups.items():
            st.markdown(f"**{k}** – {v['formation']}")
            terrain_visuel(v['formation'], v['details'])

elif menu == "Matchs":
    st.title("Matchs")
    adversaire = st.text_input("Adversaire")
    date = st.date_input("Date", value=datetime.today())
    heure = st.time_input("Heure")
    lieu = st.text_input("Lieu")
    compo = st.selectbox("Composition", [""] + list(st.session_state.lineups.keys()))
    if st.button("Enregistrer le match") and compo:
        details = st.session_state.lineups[compo]["details"]
        formation = st.session_state.lineups[compo]["formation"]
        tous_titulaires = [j for p in details for j in details[p] if j]
        remps = remplaçants_interactif("new_match", tous_titulaires)
        st.session_state.matches[f"{date} vs {adversaire}"] = {
            "adversaire": adversaire,
            "date": str(date),
            "heure": str(heure),
            "lieu": lieu,
            "formation": formation,
            "details": details,
            "remplacants": remps,
            "events": {},
            "homme_du_match": ""
        }
        save_all()
        st.success("Match ajouté")
    st.subheader("Matchs enregistrés")
    for k, m in st.session_state.matches.items():
        st.markdown(f"### {k}")
        st.write(m)

elif menu == "Export JSON":
    st.title("Export des données")
    st.download_button("Télécharger toutes les données", json.dumps({
        "players": st.session_state.players.to_dict(orient="records"),
        "lineups": st.session_state.lineups,
        "matches": st.session_state.matches
    }, indent=2), file_name="afcdata.json", mime="application/json")
