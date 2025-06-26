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

FORMATION = {
    "4-4-2": {"G": 1, "D": 4, "M": 4, "A": 2},
    "4-3-3": {"G": 1, "D": 4, "M": 3, "A": 3},
    "3-5-2": {"G": 1, "D": 3, "M": 5, "A": 2},
    "3-4-3": {"G": 1, "D": 3, "M": 4, "A": 3},
    "5-3-2": {"G": 1, "D": 5, "M": 3, "A": 2},
}

POSTES_ORDER = ["G", "D", "M", "A"]
DEFAULT_FORMATION = "4-4-2"

def afficher_terrain(formation, terrain_data=None):
    st.write("### Terrain de jeu")
    cols = st.columns(len(POSTES_ORDER))
    
    for idx, poste in enumerate(POSTES_ORDER):
        with cols[idx]:
            st.write(f"#### {poste}")
            nb_joueurs = FORMATION[formation][poste]
            for i in range(nb_joueurs):
                joueur = {}
                if terrain_data and terrain_data[poste][i]:
                    joueur = terrain_data[poste][i]
                
                with st.container():
                    st.write(f"Position {poste}{i+1}")
                    selected_player = st.selectbox(
                        "Joueur",
                        [""] + list(st.session_state.players[st.session_state.players["Poste"] == poste]["Nom"]),
                        key=f"player_{poste}_{i}",
                        index=0 if not joueur else list(st.session_state.players["Nom"]).index(joueur["nom"]) + 1
                    )
                    
                    if selected_player:
                        col1, col2 = st.columns(2)
                        with col1:
                            numero = st.number_input(
                                "Numéro",
                                min_value=1,
                                max_value=99,
                                value=joueur.get("numero", 1),
                                key=f"numero_{poste}_{i}"
                            )
                        with col2:
                            capitaine = st.checkbox(
                                "Capitaine",
                                value=joueur.get("capitaine", False),
                                key=f"capitaine_{poste}_{i}"
                            )
                        
                        terrain_data = terrain_data or {p: [None]*FORMATION[formation][p] for p in POSTES_ORDER}
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
                use_container_width=True
            )
            if st.button("Mettre à jour la liste"):
                st.session_state.players = edited_df
                st.success("Liste mise à jour!")
        else:
            st.info("Aucun joueur dans la base de données")
    
    with tabs[1]:
        st.header("Ajouter un joueur")
        with st.form("new_player"):
            nom = st.text_input("Nom")
            poste = st.selectbox("Poste", POSTES_ORDER)
            club = st.text_input("Club")
            if st.form_submit_button("Ajouter"):
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
            uploaded_file = st.file_uploader("Choisir un fichier CSV", type="csv")
            if uploaded_file is not None:
                if st.button("Charger les données"):
                    st.session_state.players = pd.read_csv(uploaded_file)
                    st.success("Données chargées!")
        
        with col2:
            st.write("### Sauvegarder")
            if st.button("Sauvegarder en CSV"):
                if "players" in st.session_state:
                    filename = st.text_input("Nom du fichier", value="players_db.csv")
                    st.session_state.players.to_csv(filename, index=False)
                    st.success(f"Données sauvegardées dans {filename}!")

def gestion_compositions():
    tabs = st.tabs(["Créer une composition", "Gérer mes compositions", "Charger/Sauvegarder"])
    
    with tabs[0]:
        st.header("Créer une composition")
        nom_compo = st.text_input("Nom de la composition")
        formation = st.selectbox("Formation", list(FORMATION.keys()), index=0)
        
        if nom_compo and formation:
            terrain_data = afficher_terrain(formation)
            
            st.subheader("Remplaçants")
            selected_players = set(
                player["nom"] for poste in terrain_data.values() 
                for player in poste if player
            ) if terrain_data else set()
            
            if "players" in st.session_state:
                available_subs = st.session_state.players[
                    ~st.session_state.players["Nom"].isin(selected_players)
                ]
                subs = st.multiselect(
                    "Sélectionner les remplaçants",
                    available_subs["Nom"]
                )
                
                if st.button("Enregistrer la composition"):
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
                list(st.session_state.lineups.keys())
            )
            
            if compo_to_edit:
                compo = st.session_state.lineups[compo_to_edit]
                st.write(f"Formation: {compo['formation']}")
                
                terrain_data = afficher_terrain(
                    compo['formation'],
                    compo['terrain']
                )
                
                st.write("### Remplaçants")
                st.write(", ".join(compo['remplacants']))
                
                if st.button("Supprimer cette composition"):
                    del st.session_state.lineups[compo_to_edit]
                    st.success(f"Composition {compo_to_edit} supprimée!")
                    st.rerun()
    
    with tabs[2]:
        st.header("Charger/Sauvegarder")
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### Charger")
            uploaded_file = st.file_uploader("Choisir un fichier JSON", type="json")
            if uploaded_file is not None:
                if st.button("Charger les compositions"):
                    st.session_state.lineups = json.load(uploaded_file)
                    st.success("Compositions chargées!")
        
        with col2:
            st.write("### Sauvegarder")
            if st.button("Sauvegarder en JSON"):
                if "lineups" in st.session_state:
                    filename = st.text_input("Nom du fichier", value="lineups.json")
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
        
        match_name = st.text_input("Nom du match", value=suggested_name)
        
        if "lineups" in st.session_state and st.session_state.lineups:
            compo = st.selectbox(
                "Sélectionner une composition",
                list(st.session_state.lineups.keys())
            )
            
            if st.button("Créer le match"):
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
                list(st.session_state.matches.keys())
            )
            
            if match_to_edit:
                match = st.session_state.matches[match_to_edit]
                st.write(f"Composition utilisée: {match['composition']}")
                
                if st.button("Supprimer ce match"):
                    # Mise à jour des statistiques
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
            uploaded_file = st.file_uploader("Choisir un fichier JSON", type="json")
            if uploaded_file is not None:
                if st.button("Charger les matchs"):
                    st.session_state.matches = json.load(uploaded_file)
                    st.success("Matchs chargés!")
        
        with col2:
            st.write("### Sauvegarder")
            if st.button("Sauvegarder en JSON"):
                if "matches" in st.session_state:
                    filename = st.text_input("Nom du fichier", value="matches.json")
                    with open(filename, "w") as f:
                        json.dump(st.session_state.matches, f, indent=2)
                    st.success(f"Matchs sauvegardés dans {filename}!")

def main():
    st.title("AFC Manager")
    
    # Menus principaux sous forme d'onglets
    main_tabs = st.tabs(["Joueurs", "Compositions", "Matchs"])
    
    with main_tabs[0]:
        gestion_joueurs()
    
    with main_tabs[1]:
        gestion_compositions()
    
    with main_tabs[2]:
        gestion_matchs()

if __name__ == "__main__":
    main()