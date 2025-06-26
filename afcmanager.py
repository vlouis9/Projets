import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

# --- Fichiers de persistance ---
DB_FILE = "players_db.csv"
LINEUPS_FILE = "lineups.json"
MATCHES_FILE = "matches.json"

PLAYER_COLS = [
    "Nom", "Poste", "Club", "Titulaire", "Infos",
    "Buts", "Passes décisives", "Cartons jaunes", "Cartons rouges",
    "Sélections", "Titularisations", "Note générale", "Homme du match"
]
PLAYER_DEFAULTS = {
    "Nom": "",
    "Poste": "G",
    "Club": "",
    "Titulaire": True,
    "Infos": "",
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

# --- Fonctions utilitaires ---
def reload_players():
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        for col in PLAYER_COLS:
            if col not in df.columns:
                df[col] = [PLAYER_DEFAULTS[col]] * len(df)
        st.session_state.players = df[PLAYER_COLS]
    else:
        st.session_state.players = pd.DataFrame(columns=PLAYER_COLS)

def save_players():
    st.session_state.players.to_csv(DB_FILE, index=False)

def reload_lineups():
    if os.path.exists(LINEUPS_FILE):
        with open(LINEUPS_FILE, "r") as f:
            st.session_state.lineups = json.load(f)
    else:
        st.session_state.lineups = {}

def save_lineups():
    with open(LINEUPS_FILE, "w") as f:
        json.dump(st.session_state.lineups, f, indent=2)

def reload_matches():
    if os.path.exists(MATCHES_FILE):
        with open(MATCHES_FILE, "r") as f:
            st.session_state.matches = json.load(f)
    else:
        st.session_state.matches = {}

def save_matches():
    with open(MATCHES_FILE, "w") as f:
        json.dump(st.session_state.matches, f, indent=2)

def terrain_init(formation):
    return {poste: [None for _ in range(FORMATION[formation][poste])] for poste in POSTES_ORDER}

# --- Gestion des compositions ---
def gestion_compositions():
    st.sidebar.header("Menu Compositions")
    compo_choice = st.sidebar.radio(
        "Choisir une action",
        ["Créer une composition", "Gérer mes compositions"]
    )
    
    if compo_choice == "Créer une composition":
        creer_composition()
    else:
        gerer_compositions()

def creer_composition():
    st.header("Créer une composition")
    
    # Charger les données
    if "lineups" not in st.session_state:
        reload_lineups()
    
    # Interface de création
    nom_compo = st.text_input("Nom de la composition")
    formation = st.selectbox("Formation", list(FORMATION.keys()), index=0)
    
    if nom_compo and formation:
        terrain = terrain_init(formation)
        selected_players = set()  # Pour suivre les joueurs déjà sélectionnés
        
        for poste in POSTES_ORDER:
            st.subheader(f"Sélection {poste}")
            available_players = st.session_state.players[
                (st.session_state.players["Poste"] == poste) & 
                (~st.session_state.players["Nom"].isin(selected_players))
            ]
            
            for i in range(FORMATION[formation][poste]):
                player = st.selectbox(
                    f"{poste} {i+1}",
                    [""] + list(available_players["Nom"]),
                    key=f"{poste}_{i}"
                )
                if player:
                    terrain[poste][i] = player
                    selected_players.add(player)
        
        # Sélection des remplaçants
        st.subheader("Remplaçants")
        available_subs = st.session_state.players[
            ~st.session_state.players["Nom"].isin(selected_players)
        ]
        subs = st.multiselect(
            "Sélectionner les remplaçants",
            available_subs["Nom"]
        )
        
        if st.button("Enregistrer la composition"):
            st.session_state.lineups[nom_compo] = {
                "formation": formation,
                "terrain": terrain,
                "remplacants": subs
            }
            save_lineups()
            st.success(f"Composition {nom_compo} enregistrée!")

def gerer_compositions():
    st.header("Gérer mes compositions")
    
    # Charger les données
    if "lineups" not in st.session_state:
        reload_lineups()
    
    if st.session_state.lineups:
        compo_to_edit = st.selectbox(
            "Sélectionner une composition",
            list(st.session_state.lineups.keys())
        )
        
        if st.button("Supprimer"):
            del st.session_state.lineups[compo_to_edit]
            save_lineups()
            st.success(f"Composition {compo_to_edit} supprimée!")

# --- Gestion des matchs ---
def gestion_matchs():
    st.sidebar.header("Menu Matchs")
    match_choice = st.sidebar.radio(
        "Choisir une action",
        ["Créer un match", "Gérer mes matchs"]
    )
    
    if match_choice == "Créer un match":
        creer_match()
    else:
        gerer_matchs()

def generer_nom_match():
    date = datetime.now().strftime("%Y-%m-%d")
    count = sum(1 for match in st.session_state.matches.keys() if date in match)
    return f"Match {date}-{count+1}"

def creer_match():
    st.header("Créer un match")
    
    # Charger les données
    if "matches" not in st.session_state:
        reload_matches()
    if "lineups" not in st.session_state:
        reload_lineups()
    
    # Suggestion automatique du nom
    suggested_name = generer_nom_match()
    match_name = st.text_input("Nom du match", value=suggested_name)
    
    if st.session_state.lineups:
        compo = st.selectbox(
            "Sélectionner une composition",
            list(st.session_state.lineups.keys())
        )
        
        if st.button("Créer le match"):
            st.session_state.matches[match_name] = {
                "composition": compo,
                "stats": {}  # Pour les statistiques du match
            }
            save_matches()
            st.success(f"Match {match_name} créé!")
    else:
        st.warning("Créez d'abord une composition!")

def gerer_matchs():
    st.header("Gérer mes matchs")
    
    # Charger les données
    if "matches" not in st.session_state:
        reload_matches()
    
    if st.session_state.matches:
        match_to_edit = st.selectbox(
            "Sélectionner un match",
            list(st.session_state.matches.keys())
        )
        
        if st.button("Supprimer"):
            # Mettre à jour les statistiques des joueurs
            match_compo = st.session_state.matches[match_to_edit]["composition"]
            match_stats = st.session_state.matches[match_to_edit].get("stats", {})
            
            # Réinitialiser les stats des joueurs pour ce match
            players_df = st.session_state.players.copy()
            for player, stats in match_stats.items():
                for stat, value in stats.items():
                    if stat in players_df.columns:
                        idx = players_df.index[players_df["Nom"] == player].tolist()
                        if idx:
                            players_df.at[idx[0], stat] -= value
            
            st.session_state.players = players_df
            save_players()
            
            # Supprimer le match
            del st.session_state.matches[match_to_edit]
            save_matches()
            st.success(f"Match {match_to_edit} supprimé et statistiques mises à jour!")

# --- Main ---
def main():
    st.title("AFC Manager")
    
    # Initialisation des données
    if "players" not in st.session_state:
        reload_players()
    
    menu = st.sidebar.selectbox(
        "Menu principal",
        ["Joueurs", "Compositions", "Matchs"]
    )
    
    if menu == "Compositions":
        gestion_compositions()
    elif menu == "Matchs":
        gestion_matchs()
    elif menu == "Joueurs":
        # Ajoutez ici la gestion des joueurs
        pass

if __name__ == "__main__":
    main()