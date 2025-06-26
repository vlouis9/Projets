import streamlit as st
import pandas as pd
import json
import os

# ----- Config fichiers -----
DB_FILE = "players_db.csv"
LINEUPS_FILE = "lineups.json"

# ----- Initialisation -----
if "players" not in st.session_state:
    if os.path.exists(DB_FILE):
        st.session_state.players = pd.read_csv(DB_FILE)
    else:
        st.session_state.players = pd.DataFrame(columns=["Nom", "Poste", "Club", "Titulaire", "Infos"])

if "lineups" not in st.session_state:
    if os.path.exists(LINEUPS_FILE):
        with open(LINEUPS_FILE, "r") as f:
            st.session_state.lineups = json.load(f)
    else:
        st.session_state.lineups = {}

# ----- Fonctions -----
def save_players():
    st.session_state.players.to_csv(DB_FILE, index=False)

def save_lineups():
    with open(LINEUPS_FILE, "w") as f:
        json.dump(st.session_state.lineups, f, indent=2)

# ----- Sidebar : Navigation -----
st.sidebar.title("⚽ Gestion Équipe")
page = st.sidebar.radio("Menu", ["Base Joueurs", "Créer Composition", "Mes Compos"])

# ----- Page Base Joueurs -----
if page == "Base Joueurs":
    st.title("📋 Base de données Joueurs")

    with st.form("add_player"):
        nom = st.text_input("Nom")
        poste = st.selectbox("Poste", ["G", "D", "M", "A"])
        club = st.text_input("Club")
        titulaire = st.checkbox("Titulaire probable", value=True)
        infos = st.text_input("Infos complémentaires")
        submitted = st.form_submit_button("Ajouter Joueur")

        if submitted and nom:
            st.session_state.players = pd.concat([
                st.session_state.players,
                pd.DataFrame([[nom, poste, club, titulaire, infos]], columns=st.session_state.players.columns)
            ], ignore_index=True)
            save_players()
            st.success(f"{nom} ajouté à la base.")

    st.write("### Joueurs enregistrés")
    st.dataframe(st.session_state.players)

# ----- Page Créer Composition -----
elif page == "Créer Composition":
    st.title("🎯 Création d'une Composition")

    nom_compo = st.text_input("Nom de la Composition")
    formation = st.selectbox("Formation", ["4-4-2", "4-3-3", "3-5-2", "3-4-3", "5-3-2"])

    lineup = {}

    for poste, nb in zip(["G", "D", "M", "A"], [1, int(formation[0]), int(formation[2]), int(formation[4])]):
        st.write(f"**{poste} - {nb} joueurs**")
        options = st.session_state.players[st.session_state.players["Poste"] == poste]["Nom"].tolist()
        selection = st.multiselect(f"Sélectionnez {nb} {poste}", options, key=poste)

        if len(selection) != nb:
            st.warning(f"Il faut sélectionner exactement {nb} joueurs pour le poste {poste}")
        lineup[poste] = selection

    if st.button("Enregistrer Composition"):
        if all(len(lineup[p]) == n for p, n in zip(["G", "D", "M", "A"], [1, int(formation[0]), int(formation[2]), int(formation[4])])):
            st.session_state.lineups[nom_compo] = {"formation": formation, "lineup": lineup}
            save_lineups()
            st.success("Composition enregistrée.")
        else:
            st.error("Merci de respecter le nombre de joueurs par poste.")

# ----- Page Visualisation Compos -----
elif page == "Mes Compos":
    st.title("📦 Mes Compositions")

    if not st.session_state.lineups:
        st.info("Aucune composition enregistrée.")
    else:
        for nom, compo in st.session_state.lineups.items():
            with st.expander(f"{nom} - {compo['formation']}"):
                for poste in ["G", "D", "M", "A"]:
                    st.write(f"**{poste}** : {', '.join(compo['lineup'].get(poste, []))}")

                if st.button(f"Supprimer {nom}", key=nom):
                    del st.session_state.lineups[nom]
                    save_lineups()
                    st.experimental_rerun()
