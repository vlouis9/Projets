import streamlit as st
import pandas as pd
import json
import os
import copy
import traceback
from datetime import datetime
import plotly.graph_objects as go

DATA_FILE = "afcdata.json"
PLAYER_COLS = ["Nom", "Poste", "Infos"]

POSTES_DETAILES = {
    "4-2-3-1": [
        ("Gardien", "G"),
        ("Défenseur gauche", "DL"),
        ("Défenseur central gauche", "DCG"),
        ("Défenseur central droit", "DCD"),
        ("Défenseur droit", "DR"),
        ("Milieu récupérateur gauche", "MRG"),
        ("Milieu récupérateur droit", "MRD"),
        ("Milieu offensif gauche", "MOG"),
        ("Milieu offensif axial", "MOA"),
        ("Milieu offensif droit", "MOD"),
        ("Attaquant", "AT"),
    ],
    "4-4-2": [
        ("Gardien", "G"),
        ("Défenseur gauche", "DL"),
        ("Défenseur central gauche", "DCG"),
        ("Défenseur central droit", "DCD"),
        ("Défenseur droit", "DR"),
        ("Milieu gauche", "MLG"),
        ("Milieu central gauche", "MCG"),
        ("Milieu central droit", "MCD"),
        ("Milieu droit", "MLD"),
        ("Attaquant gauche", "ATG"),
        ("Attaquant droit", "ATD"),
    ],
    "4-3-3": [
        ("Gardien", "G"),
        ("Défenseur gauche", "DL"),
        ("Défenseur central gauche", "DCG"),
        ("Défenseur central droit", "DCD"),
        ("Défenseur droit", "DR"),
        ("Milieu gauche", "MLG"),
        ("Milieu axial", "MA"),
        ("Milieu droit", "MLD"),
        ("Ailier gauche", "AIG"),
        ("Avant-centre", "AC"),
        ("Ailier droit", "AID"),
    ],
    "3-5-2": [
        ("Gardien", "G"),
        ("Défenseur gauche", "DL"),
        ("Défenseur central", "DC"),
        ("Défenseur droit", "DR"),
        ("Milieu gauche", "MLG"),
        ("Milieu central gauche", "MCG"),
        ("Milieu axial", "MA"),
        ("Milieu central droit", "MCD"),
        ("Milieu droit", "MLD"),
        ("Attaquant gauche", "ATG"),
        ("Attaquant droit", "ATD"),
    ],
    "3-4-3": [
        ("Gardien", "G"),
        ("Défenseur gauche", "DL"),
        ("Défenseur central", "DC"),
        ("Défenseur droit", "DR"),
        ("Milieu gauche", "MLG"),
        ("Milieu central gauche", "MCG"),
        ("Milieu central droit", "MCD"),
        ("Milieu droit", "MLD"),
        ("Ailier gauche", "AIG"),
        ("Avant-centre", "AC"),
        ("Ailier droit", "AID"),
    ],
    "5-3-2": [
        ("Gardien", "G"),
        ("Latéral gauche", "LG"),
        ("Défenseur central gauche", "DCG"),
        ("Défenseur central", "DC"),
        ("Défenseur central droit", "DCD"),
        ("Latéral droit", "LD"),
        ("Milieu gauche", "MLG"),
        ("Milieu axial", "MA"),
        ("Milieu droit", "MLD"),
        ("Attaquant gauche", "ATG"),
        ("Attaquant droit", "ATD"),
    ],
}
DEFAULT_FORMATION = "4-2-3-1"
MAX_REMPLACANTS = 5

POSITIONS_DETAILLEES = {
    "4-2-3-1": [
        (34, 8), (10, 22), (22, 22), (46, 22), (58, 22),
        (18, 40), (50, 40),
        (10, 60), (34, 60), (58, 60), (34, 88)
    ],
    "4-4-2": [
        (34, 8), (10, 22), (22, 22), (46, 22), (58, 22),
        (10, 48), (22, 48), (46, 48), (58, 48), (24, 85), (44, 85)
    ],
    "4-3-3": [
        (34, 8), (10, 22), (22, 22), (46, 22), (58, 22),
        (18, 46), (34, 52), (50, 46), (18, 80), (34, 92), (50, 80)
    ],
    "3-5-2": [
        (34, 8), (13, 18), (34, 18), (55, 18),
        (10, 40), (22, 50), (34, 60), (46, 50), (58, 40), (24, 88), (44, 88)
    ],
    "3-4-3": [
        (34, 8), (13, 18), (34, 18), (55, 18),
        (10, 46), (22, 56), (46, 56), (58, 46), (18, 80), (34, 92), (50, 80)
    ],
    "5-3-2": [
        (34, 8), (7, 20), (18, 20), (34, 20), (50, 20), (61, 20),
        (15, 52), (34, 60), (53, 52), (24, 88), (44, 88)
    ]
}

def draw_football_pitch_vertical():
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=68, y1=105, line=dict(width=2, color="#145A32"))
    fig.add_shape(type="rect", x0=13.84, y0=0, x1=68-13.84, y1=16.5, line=dict(width=1, color="#145A32"))
    fig.add_shape(type="rect", x0=13.84, y0=105-16.5, x1=68-13.84, y1=105, line=dict(width=1, color="#145A32"))
    fig.add_shape(type="circle", x0=34-9.15, y0=52.5-9.15, x1=34+9.15, y1=52.5+9.15, line=dict(width=1, color="#145A32"))
    fig.add_shape(type="circle", x0=34-0.4, y0=52.5-0.4, x1=34+0.4, y1=52.5+0.4, fillcolor="#145A32", line=dict(color="#145A32"))
    fig.update_xaxes(showticklabels=False, range=[-5, 73], visible=False)
    fig.update_yaxes(showticklabels=False, range=[-8, 125], visible=False)
    fig.update_layout(
        width=460, height=800, plot_bgcolor="#154734", margin=dict(l=10,r=10,t=10,b=10), showlegend=False
    )
    return fig

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

def terrain_interactif_detaillé(formation, terrain_key):
    postes = POSTES_DETAILES[formation]
    if terrain_key not in st.session_state:
        st.session_state[terrain_key] = {abbr: None for _, abbr in postes}
    terrain = st.session_state[terrain_key]
    for _, abbr in postes:
        if abbr not in terrain:
            terrain[abbr] = None
    col1, col2 = st.columns(2)
    all_selected = [v["Nom"] for v in terrain.values() if isinstance(v, dict) and v.get("Nom")]
    for idx, (poste_label, abbr) in enumerate(postes):
        col = col1 if idx % 2 == 0 else col2
        current_joueur = terrain[abbr] if isinstance(terrain[abbr], dict) and "Nom" in terrain[abbr] else None
        current_nom = current_joueur["Nom"] if current_joueur else ""
        joueur_options = [""] + [
            n for n in st.session_state.players["Nom"]
            if n and (n == current_nom or n not in all_selected)
        ]
        choix = col.selectbox(
            poste_label,
            joueur_options,
            index=joueur_options.index(current_nom) if current_nom in joueur_options else 0,
            key=f"{terrain_key}_{abbr}"
        )
        if choix:
            joueur_info = st.session_state.players[st.session_state.players["Nom"] == choix].iloc[0].to_dict()
            num = col.text_input(f"Numéro de {choix}", value=current_joueur.get("Numero","") if current_joueur else "", key=f"num_{terrain_key}_{abbr}")
            joueur_info["Numero"] = num
            terrain[abbr] = joueur_info
        else:
            terrain[abbr] = None
        all_selected = [v["Nom"] for v in terrain.values() if isinstance(v, dict) and v.get("Nom")]
    st.session_state[terrain_key] = terrain
    return terrain

def remplaçants_interactif(key, titulaires):
    if f"remp_{key}" not in st.session_state:
        st.session_state[f"remp_{key}"] = [{"Nom":None, "Numero":""} for _ in range(MAX_REMPLACANTS)]
    remps = st.session_state[f"remp_{key}"]
    dispo = [n for n in st.session_state.players["Nom"] if n not in [r["Nom"] for r in remps] and n not in titulaires if n]
    for i in range(MAX_REMPLACANTS):
        current = remps[i]
        options = dispo + ([current["Nom"]] if current["Nom"] and current["Nom"] not in dispo else [])
        choix = st.selectbox(
            f"Remplaçant {i+1}",
            [""] + options,
            index=(options.index(current["Nom"])+1) if current["Nom"] in options else 0,
            key=f"remp_choice_{key}_{i}"
        )
        num = st.text_input(f"Numéro Remplaçant {i+1}", value=current.get("Numero",""), key=f"num_remp_{key}_{i}")
        remps[i] = {"Nom": choix if choix else None, "Numero": num}
        dispo = [n for n in dispo if n != choix]
    st.session_state[f"remp_{key}"] = remps
    return [r for r in remps if r["Nom"]]

def plot_lineup_on_pitch_vertical(fig, details, formation, remplaçants=None, capitaine=None):
    postes = POSTES_DETAILES[formation]
    positions = POSITIONS_DETAILLEES[formation]
    for idx, ((poste_label, abbr), (x, y)) in enumerate(zip(postes, positions)):
        joueur = details.get(abbr)
        if joueur and isinstance(joueur, dict) and "Nom" in joueur:
            color = "#ffd700" if capitaine and (joueur["Nom"] == capitaine) else "#0d47a1"
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(size=38, color=color, line=dict(width=2, color="white")),
                text=f"{joueur.get('Numero', '')}".strip() if "Numero" in joueur else "",
                textposition="middle center",
                textfont=dict(color="white", size=17, family="Arial Black"),
                hovertext=f"{joueur['Nom']}"+(" (C)" if capitaine and joueur["Nom"] == capitaine else ""),
                hoverinfo="text"
            ))
            fig.add_trace(go.Scatter(
                x=[x], y=[y-4],
                mode="text",
                text=[joueur['Nom'] + (" (C)" if capitaine and joueur["Nom"] == capitaine else "")],
                textfont=dict(color="white", size=13, family="Arial Black"),
                showlegend=False
            ))
    remplaçants = remplaçants or []
    nremp = len(remplaçants)
    if nremp:
        x_start = 34 - 16*nremp/2 + 8
        for idx, remp in enumerate(remplaçants):
            x_r = x_start + idx*16
            if isinstance(remp, dict):
                nom = remp.get("Nom", "")
                numero = remp.get("Numero", "")
            else:
                nom = remp
                numero = ""
            fig.add_trace(go.Scatter(
                x=[x_r], y=[-6],
                mode="markers+text",
                marker=dict(size=28, color="#0d47a1", line=dict(width=2, color="white")),
                text=f"{numero}",
                textposition="middle center",
                textfont=dict(color="white", size=13, family="Arial Black"),
                hovertext=nom,
                hoverinfo="text"
            ))
            fig.add_trace(go.Scatter(
                x=[x_r], y=[-11],
                mode="text",
                text=[nom],
                textfont=dict(color="white", size=12, family="Arial Black"),
                showlegend=False
            ))
    return fig

# === Interface Streamlit ===

st.set_page_config(layout="wide")
if "players" not in st.session_state:
    reload_all()
if "lineups" not in st.session_state:
    st.session_state.lineups = {}
if "matches" not in st.session_state:
    st.session_state.matches = {}

st.title("AFC Manager - Gestion d'équipe et compos")

menu = st.sidebar.radio("Menu", [
    "Joueurs", "Compositions", "Matchs", "Stats", "Import/Export"
])

if menu == "Joueurs":
    st.header("Gestion des joueurs")
    df = st.session_state.players
    st.dataframe(df)
    with st.form("Ajouter joueur"):
        nom = st.text_input("Nom")
        poste = st.text_input("Poste")
        infos = st.text_input("Infos")
        if st.form_submit_button("Ajouter"):
            if nom and poste:
                st.session_state.players = pd.concat([df, pd.DataFrame([{"Nom": nom, "Poste": poste, "Infos": infos}])], ignore_index=True)
                save_all()
                st.success("Ajouté !")
            else:
                st.error("Nom et poste obligatoires.")
    if st.button("Vider tous les joueurs", key="clear_players", help="Efface définitivement tous les joueurs !"):
        st.session_state.players = pd.DataFrame(columns=PLAYER_COLS)
        save_all()
        st.success("Joueurs effacés.")

elif menu == "Compositions":
    st.header("Création et gestion des compositions")
    noms_lineups = list(st.session_state.lineups.keys())
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Nouvelle composition"):
            st.session_state.new_lineup = True
        choix_lineup = st.selectbox("Charger une composition existante :", [""] + noms_lineups, key="choix_lineup")
        if choix_lineup:
            st.session_state.lineup_name = choix_lineup
            st.session_state.lineup_formation = st.session_state.lineups[choix_lineup]["formation"]
            st.session_state["terrain_edit"] = copy.deepcopy(st.session_state.lineups[choix_lineup]["titulaire"])
            st.session_state["remp_edit"] = copy.deepcopy(st.session_state.lineups[choix_lineup]["remplaçants"])
            st.session_state["capitaine_edit"] = st.session_state.lineups[choix_lineup].get("capitaine")
    with col2:
        if st.button("Supprimer cette composition") and st.session_state.get("lineup_name") in noms_lineups:
            del st.session_state.lineups[st.session_state["lineup_name"]]
            save_all()
            st.success("Composition supprimée.")
            st.experimental_rerun()
    if st.session_state.get("new_lineup", False) or st.session_state.get("lineup_name"):
        formation = st.selectbox("Choisissez la formation", list(POSTES_DETAILES.keys()), key="formation_compo",
                                index=list(POSTES_DETAILES.keys()).index(st.session_state.get("lineup_formation", DEFAULT_FORMATION)))
        terrain = terrain_interactif_detaillé(formation, "terrain_edit")
        titulaires = [v["Nom"] for v in terrain.values() if isinstance(v, dict) and v.get("Nom")]
        remplaçants = remplaçants_interactif("edit", titulaires)
        capitaine = st.selectbox("Capitaine", [""] + titulaires, key="capitaine_edit",
                                 index=titulaires.index(st.session_state.get("capitaine_edit"))+1 if st.session_state.get("capitaine_edit") in titulaires else 0)
        fig = draw_football_pitch_vertical()
        fig = plot_lineup_on_pitch_vertical(fig, terrain, formation, remplaçants, capitaine)
        st.plotly_chart(fig, use_container_width=True)
        nom_lineup = st.text_input("Nom de la composition", value=st.session_state.get("lineup_name", ""))
        if st.button("Enregistrer cette composition"):
            try:
                if not nom_lineup:
                    st.error("Le nom de la composition est obligatoire !")
                elif len(titulaires) != len(POSTES_DETAILES[formation]):
                    st.error("Il faut autant de titulaires que de postes.")
                else:
                    st.session_state.lineups[nom_lineup] = {
                        "formation": formation,
                        "titulaire": copy.deepcopy(terrain),
                        "remplaçants": copy.deepcopy(remplaçants),
                        "capitaine": capitaine
                    }
                    st.session_state.lineup_name = nom_lineup
                    st.session_state.lineup_formation = formation
                    save_all()
                    st.success("Composition sauvegardée.")
                    st.session_state.new_lineup = False
            except Exception as e:
                st.error(f"Erreur lors de l'enregistrement : {e}\n{traceback.format_exc()}")

elif menu == "Matchs":
    st.header("Gestion des matchs")
    noms_matchs = list(st.session_state.matches.keys())
    with st.expander("Nouveau match"):
        nom_match = st.text_input("Nom du match (date/adversaire)")
        choix_compo = st.selectbox("Composition", [""]+list(st.session_state.lineups.keys()))
        score_n = st.text_input("Score équipe")
        score_a = st.text_input("Score adversaire")
        if st.button("Créer le match"):
            if not nom_match or not choix_compo:
                st.warning("Nom du match et composition obligatoires.")
            else:
                st.session_state.matches[nom_match] = {
                    "composition": choix_compo,
                    "score": f"{score_n} - {score_a}",
                    "feuille": {}
                }
                save_all()
                st.success("Match créé.")
    choix_match = st.selectbox("Match à afficher", [""]+noms_matchs, key="aff_match")
    if choix_match:
        infos = st.session_state.matches[choix_match]
        lineup = st.session_state.lineups.get(infos["composition"], {})
        formation = lineup.get("formation", DEFAULT_FORMATION)
        terrain = lineup.get("titulaire", {})
        remplaçants = lineup.get("remplaçants", [])
        capitaine = lineup.get("capitaine")
        st.write(f"Score : {infos.get('score', '')}")
        fig = draw_football_pitch_vertical()
        fig = plot_lineup_on_pitch_vertical(fig, terrain, formation, remplaçants, capitaine)
        st.plotly_chart(fig, use_container_width=True)

elif menu == "Stats":
    st.header("Statistiques individuelles")
    joueurs = st.session_state.players["Nom"].tolist()
    data = []
    for nom, lineup in st.session_state.lineups.items():
        for abbr, joueur in lineup["titulaire"].items():
            if isinstance(joueur, dict) and joueur.get("Nom"):
                data.append({"Compo": nom, "Joueur": joueur["Nom"], "Poste": abbr})
        for remp in lineup.get("remplaçants", []):
            if isinstance(remp, dict) and remp.get("Nom"):
                data.append({"Compo": nom, "Joueur": remp["Nom"], "Poste": "Remplaçant"})
            elif isinstance(remp, str):
                data.append({"Compo": nom, "Joueur": remp, "Poste": "Remplaçant"})
    df_stats = pd.DataFrame(data)
    st.dataframe(df_stats.value_counts(["Joueur", "Poste"]).reset_index(name="Sélections"))

elif menu == "Import/Export":
    st.header("Importer/Exporter les données")
    st.download_button("Exporter les données", json.dumps({
        "players": st.session_state.players.to_dict(orient="records"),
        "lineups": st.session_state.lineups,
        "matches": st.session_state.matches,
    }, indent=2), file_name="afcdata_export.json")
    fichier = st.file_uploader("Importer un fichier json", type="json")
    if fichier and st.button("Importer ce fichier"):
        data = json.load(fichier)
        st.session_state.players = pd.DataFrame(data.get("players", []))
        st.session_state.lineups = data.get("lineups", {})
        st.session_state.matches = data.get("matches", {})
        save_all()
        st.success("Importé !")