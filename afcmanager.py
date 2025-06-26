import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

DB_FILE = "players_db.csv"
LINEUPS_FILE = "lineups.json"
MATCHES_FILE = "matches.json"

# ----- Initialisation -----
player_cols = [
    "Nom", "Poste", "Club", "Titulaire", "Infos", 
    "Buts", "Passes décisives", "Cartons jaunes", "Cartons rouges",
    "Sélections", "Titularisations", "Note générale"
]

def init_players():
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        # Ajout colonnes si manquantes
        for col in player_cols:
            if col not in df.columns:
                df[col] = 0 if col not in ["Nom", "Poste", "Club", "Titulaire", "Infos"] else ""
        return df[player_cols]
    else:
        return pd.DataFrame(columns=player_cols)

if "players" not in st.session_state:
    st.session_state.players = init_players()

if "lineups" not in st.session_state:
    if os.path.exists(LINEUPS_FILE):
        with open(LINEUPS_FILE, "r") as f:
            st.session_state.lineups = json.load(f)
    else:
        st.session_state.lineups = {}

if "matches" not in st.session_state:
    if os.path.exists(MATCHES_FILE):
        with open(MATCHES_FILE, "r") as f:
            st.session_state.matches = json.load(f)
    else:
        st.session_state.matches = {}

def save_players():
    st.session_state.players.to_csv(DB_FILE, index=False)

def save_lineups():
    with open(LINEUPS_FILE, "w") as f:
        json.dump(st.session_state.lineups, f, indent=2)

def save_matches():
    with open(MATCHES_FILE, "w") as f:
        json.dump(st.session_state.matches, f, indent=2)

# ----- Sidebar -----
st.sidebar.title("⚽ Gestion Équipe")
page = st.sidebar.radio("Menu", ["Base Joueurs", "Créer Composition", "Mes Compos", "Matchs"])

# ----- Helper : visuel terrain -----
import matplotlib.pyplot as plt

def draw_field(formation, lineup_details):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    # Terrain
    rect = plt.Rectangle((0, 0), 100, 100, fill=False, color="green")
    ax.add_patch(rect)
    ax.axis("off")
    # Placement joueurs
    formation_map = {
        "4-4-2": [("G", [50]), ("D", [15,35,65,85]), ("M", [20,40,60,80]), ("A", [35,65])],
        "4-3-3": [("G", [50]), ("D", [15,35,65,85]), ("M", [30,50,70]), ("A", [20,50,80])],
        "3-5-2": [("G", [50]), ("D", [25,50,75]), ("M", [15,32,50,68,85]), ("A", [40,60])],
        "3-4-3": [("G", [50]), ("D", [25,50,75]), ("M", [25,50,75,90]), ("A", [20,50,80])],
        "5-3-2": [("G", [50]), ("D", [10,25,50,75,90]), ("M", [30,50,70]), ("A", [40,60])],
    }
    y_map = {"G":10, "D":30, "M":60, "A":85}
    for poste, xs in formation_map[formation]:
        for idx, x in enumerate(xs):
            player = ""
            numero = ""
            cap = ""
            if poste in lineup_details and idx < len(lineup_details[poste]):
                pinfo = lineup_details[poste][idx]
                player = pinfo.get("Nom", "")
                numero = f"#{pinfo.get('Numero','')}" if pinfo.get("Numero") else ""
                cap = " (C)" if pinfo.get("Capitaine") else ""
            ax.text(x, y_map[poste], f"{player} {numero}{cap}", ha='center', va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.7, boxstyle='round'))
    st.pyplot(fig)

# ----- Base Joueurs -----
if page == "Base Joueurs":
    st.title("📋 Base de données Joueurs")

    st.write("### Joueurs enregistrés")
    st.dataframe(st.session_state.players)

    with st.form("add_player"):
        nom = st.text_input("Nom")
        poste = st.selectbox("Poste", ["G", "D", "M", "A"])
        club = st.text_input("Club")
        titulaire = st.checkbox("Titulaire probable", value=True)
        infos = st.text_input("Infos complémentaires")
        # Nouvelles stats
        buts = st.number_input("Buts", min_value=0, value=0)
        pdec = st.number_input("Passes décisives", min_value=0, value=0)
        cj = st.number_input("Cartons jaunes", min_value=0, value=0)
        cr = st.number_input("Cartons rouges", min_value=0, value=0)
        selections = st.number_input("Sélections", min_value=0, value=0)
        titul = st.number_input("Titularisations", min_value=0, value=0)
        note = st.number_input("Note générale", min_value=0.0, max_value=10.0, value=0.0, step=0.1)
        submitted = st.form_submit_button("Ajouter Joueur")
        if submitted and nom:
            new_row = [nom, poste, club, titulaire, infos, buts, pdec, cj, cr, selections, titul, note]
            st.session_state.players = pd.concat([
                st.session_state.players,
                pd.DataFrame([new_row], columns=player_cols)
            ], ignore_index=True)
            save_players()
            st.success(f"{nom} ajouté à la base.")

# ----- Création Composition -----
elif page == "Créer Composition":
    st.title("🎯 Création d'une Composition")
    nom_compo = st.text_input("Nom de la Composition")
    formation = st.selectbox("Formation", ["4-4-2", "4-3-3", "3-5-2", "3-4-3", "5-3-2"])
    lineup = {}
    lineup_details = {}
    nbs = [1, int(formation[0]), int(formation[2]), int(formation[4])]
    postes = ["G", "D", "M", "A"]

    st.write("#### Sélection des joueurs et attribution des numéros/capitaine")
    for poste, nb in zip(postes, nbs):
        st.write(f"**{poste} - {nb} joueurs**")
        options = st.session_state.players[st.session_state.players["Poste"] == poste]["Nom"].tolist()
        selections = st.multiselect(f"Sélectionnez {nb} {poste}", options, key=f"sel_{poste}")
        # Saisie numéro/capitaine
        details = []
        for i in range(nb):
            nom_j = selections[i] if i < len(selections) else ""
            col1, col2, col3 = st.columns(3)
            with col1:
                joueur = st.selectbox(f"Joueur {i+1}", options, key=f"{poste}_{i}", index=options.index(nom_j) if nom_j in options else 0) if options else ""
            with col2:
                num = st.number_input(f"N° {poste}{i+1}", min_value=0, value=0, key=f"num_{poste}_{i}")
            with col3:
                cap = st.checkbox(f"Capitaine ?", key=f"cap_{poste}_{i}")
            if joueur:
                details.append({"Nom": joueur, "Numero": num, "Capitaine": cap})
        lineup[poste] = [d["Nom"] for d in details]
        lineup_details[poste] = details

    st.write("#### Aperçu de la composition sur le terrain")
    draw_field(formation, lineup_details)

    if st.button("Enregistrer Composition"):
        if all(len(lineup[p]) == n for p, n in zip(postes, nbs)):
            st.session_state.lineups[nom_compo] = {
                "formation": formation,
                "lineup": lineup,
                "details": lineup_details
            }
            save_lineups()
            st.success("Composition enregistrée.")
        else:
            st.error("Merci de respecter le nombre de joueurs par poste.")

# ----- Mes Compos -----
elif page == "Mes Compos":
    st.title("📦 Mes Compositions")
    if not st.session_state.lineups:
        st.info("Aucune composition enregistrée.")
    else:
        for nom, compo in st.session_state.lineups.items():
            with st.expander(f"{nom} - {compo['formation']}"):
                for poste in ["G", "D", "M", "A"]:
                    st.write(f"**{poste}** :")
                    for joueur in compo.get("details", {}).get(poste, []):
                        txt = f"{joueur['Nom']} (N°{joueur['Numero']})" + (" (C)" if joueur.get("Capitaine") else "")
                        st.write(txt)
                st.write("**Visuel du terrain :**")
                draw_field(compo["formation"], compo["details"])
                if st.button(f"Supprimer {nom}", key=nom):
                    del st.session_state.lineups[nom]
                    save_lineups()
                    st.experimental_rerun()

# ----- Matchs -----
elif page == "Matchs":
    st.title("🏆 Matchs")
    tab1, tab2 = st.tabs(["Créer un match", "Mes matchs"])
    # --- Création ---
    with tab1:
        st.subheader("Nouveau match")
        type_match = st.selectbox("Type de match", ["Championnat", "Coupe"])
        adversaire = st.text_input("Nom de l'adversaire")
        date = st.date_input("Date du match", value=datetime.today())
        heure = st.time_input("Heure du match")
        lieu = st.text_input("Lieu")
        use_compo = st.checkbox("Utiliser une composition existante ?")
        if use_compo and st.session_state.lineups:
            compo_choice = st.selectbox("Choisir une composition", list(st.session_state.lineups.keys()))
            compo_data = st.session_state.lineups[compo_choice]
            formation = compo_data["formation"]
            lineup_details = compo_data["details"]
        else:
            formation = st.selectbox("Formation", ["4-4-2", "4-3-3", "3-5-2", "3-4-3", "5-3-2"])
            lineup_details = {poste: [] for poste in ["G", "D", "M", "A"]}
            nbs = [1, int(formation[0]), int(formation[2]), int(formation[4])]
            for poste, nb in zip(["G", "D", "M", "A"], nbs):
                options = st.session_state.players[st.session_state.players["Poste"] == poste]["Nom"].tolist()
                selections = st.multiselect(f"{poste} ({nb})", options, key=f"match_{poste}")
                details = []
                for i in range(nb):
                    joueur = selections[i] if i < len(selections) else ""
                    num = st.number_input(f"N° {poste}{i+1}", min_value=0, value=0, key=f"matchnum_{poste}_{i}")
                    cap = st.checkbox(f"Capitaine ?", key=f"matchcap_{poste}_{i}")
                    if joueur:
                        details.append({"Nom": joueur, "Numero": num, "Capitaine": cap})
                lineup_details[poste] = details

        st.write("**Visuel de la compo**")
        draw_field(formation, lineup_details)

        # Remplaçants
        subs = st.multiselect("Remplaçants", st.session_state.players["Nom"].tolist())
        if st.button("Enregistrer Match"):
            match_id = f"{date}_{adversaire}_{heure}"
            st.session_state.matches[match_id] = {
                "type": type_match,
                "adversaire": adversaire,
                "date": str(date),
                "heure": str(heure),
                "lieu": lieu,
                "formation": formation,
                "details": lineup_details,
                "remplacants": subs,
                "events": {},
                "score": "",
                "noted": False
            }
            save_matches()
            st.success("Match enregistré.")

    # --- Consultation/Edition ---
    with tab2:
        if not st.session_state.matches:
            st.info("Aucun match enregistré.")
        else:
            for mid, match in st.session_state.matches.items():
                with st.expander(f"{match['date']} {match['heure']} vs {match['adversaire']} ({match['type']})"):
                    st.write(f"**Lieu :** {match['lieu']}")
                    st.write(f"**Formation :** {match['formation']}")
                    st.write("**Titularaires :**")
                    for poste in ["G", "D", "M", "A"]:
                        for joueur in match["details"].get(poste, []):
                            txt = f"{joueur['Nom']} (N°{joueur['Numero']})" + (" (C)" if joueur.get("Capitaine") else "")
                            st.write(txt)
                    st.write("**Remplaçants :** " + ", ".join(match.get("remplacants", [])))
                    draw_field(match["formation"], match["details"])

                    # Saisie post-match
                    if not match.get("noted", False):
                        st.write("### Saisie des stats du match")
                        score = st.text_input("Score (ex: 2-1)", key=f"score_{mid}")
                        buteurs = st.multiselect("Buteurs", st.session_state.players["Nom"].tolist(), key=f"buteurs_{mid}")
                        passeurs = st.multiselect("Passeurs", st.session_state.players["Nom"].tolist(), key=f"passeurs_{mid}")
                        cj = st.multiselect("Cartons jaunes", st.session_state.players["Nom"].tolist(), key=f"cj_{mid}")
                        cr = st.multiselect("Cartons rouges", st.session_state.players["Nom"].tolist(), key=f"cr_{mid}")
                        notes = {}
                        for nom in [j["Nom"] for p in ["G", "D", "M", "A"] for j in match["details"].get(p, [])]:
                            notes[nom] = st.number_input(f"Note pour {nom}", min_value=0.0, max_value=10.0, value=6.0, step=0.1, key=f"note_{mid}_{nom}")
                        if st.button("Valider le match", key=f"valide_{mid}"):
                            # MAJ stats joueurs
                            df = st.session_state.players
                            for nom in [j["Nom"] for p in ["G", "D", "M", "A"] for j in match["details"].get(p, [])]:
                                idx = df[df["Nom"] == nom].index
                                if len(idx):
                                    i = idx[0]
                                    df.at[i, "Sélections"] = df.at[i, "Sélections"] + 1
                                    df.at[i, "Titularisations"] = df.at[i, "Titularisations"] + 1
                                    df.at[i, "Note générale"] = round((df.at[i, "Note générale"] + notes[nom]) / 2, 2) if df.at[i, "Note générale"] > 0 else notes[nom]
                            for nom in match.get("remplacants", []):
                                idx = df[df["Nom"] == nom].index
                                if len(idx):
                                    i = idx[0]
                                    df.at[i, "Sélections"] = df.at[i, "Sélections"] + 1
                            for nom in buteurs:
                                idx = df[df["Nom"] == nom].index
                                if len(idx): df.at[idx[0], "Buts"] = df.at[idx[0], "Buts"] + 1
                            for nom in passeurs:
                                idx = df[df["Nom"] == nom].index
                                if len(idx): df.at[idx[0], "Passes décisives"] = df.at[idx[0], "Passes décisives"] + 1
                            for nom in cj:
                                idx = df[df["Nom"] == nom].index
                                if len(idx): df.at[idx[0], "Cartons jaunes"] = df.at[idx[0], "Cartons jaunes"] + 1
                            for nom in cr:
                                idx = df[df["Nom"] == nom].index
                                if len(idx): df.at[idx[0], "Cartons rouges"] = df.at[idx[0], "Cartons rouges"] + 1
                            save_players()
                            match["score"] = score
                            match["events"] = {
                                "buteurs": buteurs,
                                "passeurs": passeurs,
                                "cartons_jaunes": cj,
                                "cartons_rouges": cr,
                                "notes": notes
                            }
                            match["noted"] = True
                            save_matches()
                            st.success("Stats du match enregistrées !")
                            st.experimental_rerun()
                    else:
                        st.write(f"**Score :** {match['score']}")
                        ev = match.get("events", {})
                        st.write("**Buteurs :** " + ", ".join(ev.get("buteurs", [])))
                        st.write("**Passeurs :** " + ", ".join(ev.get("passeurs", [])))
                        st.write("**Cartons jaunes :** " + ", ".join(ev.get("cartons_jaunes", [])))
                        st.write("**Cartons rouges :** " + ", ".join(ev.get("cartons_rouges", [])))
                        st.write("**Notes :**")
                        for nom, note in ev.get("notes", {}).items():
                            st.write(f"{nom} : {note}")

                    if st.button(f"Supprimer le match", key=f"suppr_{mid}"):
                        del st.session_state.matches[mid]
                        save_matches()
                        st.experimental_rerun()
