import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

# --- Constants ---
PLAYER_COLS = [
    "Nom", "Poste", "Club", "Titulaire", "Infos", "Numéro", "Capitaine",
    "Buts", "Passes décisives", "Cartons jaunes", "Cartons rouges",
    "Sélections", "Titularisations", "Note générale", "Homme du match"
]

PLAYER_DEFAULTS = {
    "Nom": "",
    "Poste": "G",
    "Club": "",
    "Titulaire": True,
    "Infos": "",
    "Numéro": 0,
    "Capitaine": False,
    "Buts": 0,
    "Passes décisives": 0,
    "Cartons jaunes": 0,
    "Cartons rouges": 0,
    "Sélections": 0,
    "Titularisations": 0,
    "Note générale": 0.0,
    "Homme du match": 0
}

# Définition des formations
FORMATION = {
    "4-4-2": {"G": 1, "D": 4, "M": 4, "A": 2},
    "4-3-3": {"G": 1, "D": 4, "M": 3, "A": 3},
    "3-5-2": {"G": 1, "D": 3, "M": 5, "A": 2},
    "3-4-3": {"G": 1, "D": 3, "M": 4, "A": 3},
    "5-3-2": {"G": 1, "D": 5, "M": 3, "A": 2},
}

# Ordre des postes pour l'affichage
POSTES_ORDER = ["G", "D", "M", "A"]
DEFAULT_FORMATION = "4-4-2"

def afficher_terrain(formation, terrain_data=None, prefix_key=""):
    st.write("### Terrain de jeu")
    cols = st.columns(len(POSTES_ORDER))
    
    terrain_data = terrain_data or {p: [None]*FORMATION[formation][p] for p in POSTES_ORDER}
    
    for idx, poste in enumerate(POSTES_ORDER):
        with cols[idx]:
            st.write(f"#### {poste}")
            nb_joueurs = FORMATION[formation][poste]
            for i in range(nb_joueurs):
                joueur = terrain_data[poste][i] if terrain_data[poste][i] else {}
                
                with st.container():
                    st.write(f"Position {poste}{i+1}")
                    selected_player = st.selectbox(
                        "Joueur",
                        [""] + list(st.session_state.players[st.session_state.players["Poste"] == poste]["Nom"]),
                        key=f"{prefix_key}player_{poste}_{i}_{formation}",
                        index=0 if not joueur else list(st.session_state.players["Nom"]).index(joueur.get("nom", "")) + 1
                    )
                    
                    if selected_player:
                        col1, col2 = st.columns(2)
                        with col1:
                            numero = st.number_input(
                                "Numéro",
                                min_value=1,
                                max_value=99,
                                value=joueur.get("numero", 1),
                                key=f"{prefix_key}numero_{poste}_{i}_{formation}"
                            )
                        with col2:
                            capitaine = st.checkbox(
                                "Capitaine",
                                value=joueur.get("capitaine", False),
                                key=f"{prefix_key}capitaine_{poste}_{i}_{formation}"
                            )
                        
                        terrain_data[poste][i] = {
                            "nom": selected_player,
                            "numero": numero,
                            "capitaine": capitaine
                        }
    
    return terrain_data

def gestion_joueurs():
    tabs = st.tabs(["Liste des joueurs", "Ajouter un joueur", "Charger/Sauvegarder"])
    
    with tabs[0]:
        st.header("Liste des joueurs")
        if "players" in st.session_state and len(st.session_state.players) > 0:
            edited_df = st.data_editor(
                st.session_state.players,
                num_rows="dynamic",
                use_container_width=True,
                key="players_editor"
            )
            if st.button("Mettre à jour la liste", key="update_players"):
                st.session_state.players = edited_df
                st.success("Liste mise à jour!")
        else:
            st.info("Aucun joueur dans la base de données")
    
    with tabs[1]:
        st.header("Ajouter un joueur")
        nom = st.text_input("Nom", key="new_player_name")
        poste = st.selectbox("Poste", POSTES_ORDER, key="new_player_position")
        club = st.text_input("Club", key="new_player_club")
        if st.button("Ajouter", key="add_player"):
            new_player = PLAYER_DEFAULTS.copy()
            new_player.update({
                "Nom": nom,
                "Poste": poste,
                "Club": club
            })
            if "players" not in st.session_state:
                st.session_state.players = pd.DataFrame(columns=PLAYER_COLS)
            st.session_state.players = pd.concat([
                st.session_state.players,
                pd.DataFrame([new_player])
            ], ignore_index=True)
            st.success(f"Joueur {nom} ajouté!")
    
    with tabs[2]:
        st.header("Charger/Sauvegarder")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Charger")
            uploaded_file = st.file_uploader(
                "Choisir un fichier CSV",
                type="csv",
                key="players_file_uploader"
            )
            if uploaded_file is not None:
                if st.button("Charger les données", key="load_players"):
                    st.session_state.players = pd.read_csv(uploaded_file)
                    st.success("Données chargées!")
        
        with col2:
            st.write("### Sauvegarder")
            if st.button("Sauvegarder en CSV", key="save_players"):
                if "players" in st.session_state:
                    filename = st.text_input(
                        "Nom du fichier",
                        value="players_db.csv",
                        key="players_filename"
                    )
                    st.session_state.players.to_csv(filename, index=False)
                    st.success(f"Données sauvegardées dans {filename}!")

def gestion_compositions():
    tabs = st.tabs(["Créer une composition", "Gérer mes compositions", "Charger/Sauvegarder"])
    
    with tabs[0]:
        st.header("Créer une composition")
        nom_compo = st.text_input("Nom de la composition", key="new_compo_name")
        formation = st.selectbox(
            "Formation",
            list(FORMATION.keys()),
            index=0,
            key="new_compo_formation"
        )
        
        if nom_compo and formation and "players" in st.session_state:
            terrain_data = afficher_terrain(formation, prefix_key="new_")
            
            st.subheader("Remplaçants")
            selected_players = set(
                player["nom"] for poste in terrain_data.values() 
                for player in poste if player
            ) if terrain_data else set()
            
            available_subs = st.session_state.players[
                ~st.session_state.players["Nom"].isin(selected_players)
            ]
            subs = st.multiselect(
                "Sélectionner les remplaçants",
                available_subs["Nom"],
                key="new_compo_subs"
            )
            
            if st.button("Enregistrer la composition", key="save_new_compo"):
                if "lineups" not in st.session_state:
                    st.session_state.lineups = {}
                st.session_state.lineups[nom_compo] = {
                    "formation": formation,
                    "terrain": terrain_data,
                    "remplacants": subs
                }
                st.success(f"Composition {nom_compo} enregistrée!")
    
    with tabs[1]:
        st.header("Gérer mes compositions")
        if "lineups" in st.session_state and st.session_state.lineups:
            compo_to_edit = st.selectbox(
                "Sélectionner une composition",
                list(st.session_state.lineups.keys()),
                key="select_compo_edit"
            )
            
            if compo_to_edit:
                compo = st.session_state.lineups[compo_to_edit]
                st.write(f"Formation: {compo['formation']}")
                
                terrain_data = afficher_terrain(
                    compo['formation'],
                    compo['terrain'],
                    prefix_key="edit_"
                )
                
                st.write("### Remplaçants")
                st.write(", ".join(compo['remplacants']))
                
                if st.button("Supprimer cette composition", key="delete_compo"):
                    del st.session_state.lineups[compo_to_edit]
                    st.success(f"Composition {compo_to_edit} supprimée!")
                    st.rerun()
    
    with tabs[2]:
        st.header("Charger/Sauvegarder")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Charger")
            uploaded_file = st.file_uploader(
                "Choisir un fichier JSON",
                type="json",
                key="compo_file_uploader"
            )
            if uploaded_file is not None:
                if st.button("Charger les compositions", key="load_compo"):
                    st.session_state.lineups = json.load(uploaded_file)
                    st.success("Compositions chargées!")
        
        with col2:
            st.write("### Sauvegarder")
            if st.button("Sauvegarder en JSON", key="save_compo"):
                if "lineups" in st.session_state:
                    filename = st.text_input(
                        "Nom du fichier",
                        value="lineups.json",
                        key="compo_filename"
                    )
                    with open(filename, "w") as f:
                        json.dump(st.session_state.lineups, f, indent=2)
                    st.success(f"Compositions sauvegardées dans {filename}!")

def gestion_matchs():
    tabs = st.tabs(["Créer un match", "Gérer mes matchs", "Charger/Sauvegarder"])
    
    with tabs[0]:
        st.header("Créer un match")
        suggested_name = datetime.now().strftime("Match %Y-%m-%d")
        if "matches" in st.session_state:
            count = sum(1 for match in st.session_state.matches.keys() if suggested_name in match)
            suggested_name = f"{suggested_name}-{count+1}"
        
        match_name = st.text_input(
            "Nom du match",
            value=suggested_name,
            key="new_match_name"
        )
        
        if "lineups" in st.session_state and st.session_state.lineups:
            compo = st.selectbox(
                "Sélectionner une composition",
                list(st.session_state.lineups.keys()),
                key="new_match_compo"
            )
            
            if st.button("Créer le match", key="create_match"):
                if "matches" not in st.session_state:
                    st.session_state.matches = {}
                st.session_state.matches[match_name] = {
                    "composition": compo,
                    "stats": {}
                }
                st.success(f"Match {match_name} créé!")
        else:
            st.warning("Créez d'abord une composition!")
    
    with tabs[1]:
        st.header("Gérer mes matchs")
        if "matches" in st.session_state and st.session_state.matches:
            match_to_edit = st.selectbox(
                "Sélectionner un match",
                list(st.session_state.matches.keys()),
                key="select_match_edit"
            )
            
            if match_to_edit:
                match = st.session_state.matches[match_to_edit]
                st.write(f"Composition utilisée: {match['composition']}")
                
                if st.button("Supprimer ce match", key="delete_match"):
                    match_stats = match.get("stats", {})
                    if "players" in st.session_state:
                        players_df = st.session_state.players.copy()
                        for player, stats in match_stats.items():
                            for stat, value in stats.items():
                                if stat in players_df.columns:
                                    idx = players_df.index[players_df["Nom"] == player].tolist()
                                    if idx:
                                        players_df.at[idx[0], stat] -= value
                        
                        st.session_state.players = players_df
                    
                    del st.session_state.matches[match_to_edit]
                    st.success(f"Match {match_to_edit} supprimé et statistiques mises à jour!")
                    st.rerun()
    
    with tabs[2]:
        st.header("Charger/Sauvegarder")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Charger")
            uploaded_file = st.file_uploader(
                "Choisir un fichier JSON",
                type="json",
                key="match_file_uploader"
            )
            if uploaded_file is not None:
                if st.button("Charger les matchs", key="load_matches"):
                    st.session_state.matches = json.load(uploaded_file)
                    st.success("Matchs chargés!")
        
        with col2:
            st.write("### Sauvegarder")
            if st.button("Sauvegarder en JSON", key="save_matches"):
                if "matches" in st.session_state:
                    filename = st.text_input(
                        "Nom du fichier",
                        value="matches.json",
                        key="matches_filename"
                    )
                    with open(filename, "w") as f:
                        json.dump(st.session_state.matches, f, indent=2)
                    st.success(f"Matchs sauvegardés dans {filename}!")

def main():
    st.title("AFC Manager")
    
    # Initialize session state for players if not exists
    if "players" not in st.session_state:
        st.session_state.players = pd.DataFrame(columns=PLAYER_COLS)
    
    main_tabs = st.tabs(["Joueurs", "Compositions", "Matchs"])
    
    with main_tabs[0]:
        gestion_joueurs()
    
    with main_tabs[1]:
        gestion_compositions()
    
    with main_tabs[2]:
        gestion_matchs()

if __name__ == "__main__":
    main()