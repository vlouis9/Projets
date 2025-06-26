import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

# --- Constantes fichiers et structures ---
DB_FILE = "players_db.csv"
LINEUPS_FILE = "lineups.json"
MATCHES_FILE = "matches.json"

PLAYER_COLS = [
    "Nom", "Poste", "Club", "Titulaire", "Infos",
    "Buts", "Passes décisives", "Cartons jaunes", "Cartons rouges",
    "Sélections", "Titularisations", "Note générale"
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
    "Note générale": 0.0
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
    # Crée toujours la structure du terrain pour la formation donnée
    return {poste: [None for _ in range(FORMATION[formation][poste])] for poste in POSTES_ORDER}

# --- Initialisation session ---
if "players" not in st.session_state:
    reload_players()
if "lineups" not in st.session_state:
    reload_lineups()
if "matches" not in st.session_state:
    reload_matches()
if "formation" not in st.session_state:
    st.session_state.formation = DEFAULT_FORMATION

# --- Terrain interactif (fonction réutilisable) ---
def terrain_interactif(formation, terrain_key):
    """Affiche un terrain interactif, retourne le dict terrain, propose TOUS les joueurs au choix à chaque poste"""
    if terrain_key not in st.session_state or st.session_state.get(f"formation_{terrain_key}", None) != formation:
        # On réinitialise le terrain à chaque changement de formation
        st.session_state[terrain_key] = terrain_init(formation)
        st.session_state[f"formation_{terrain_key}"] = formation
    terrain = st.session_state[terrain_key]

    st.caption("Cliquez sur une case pour modifier le joueur à cette position.")

    def poste_buttons(poste, n):
        cols = st.columns(n)
        for i in range(n):
            joueur = terrain[poste][i]
            if joueur:
                label = f"{joueur['Nom']} (#{joueur['Numero']}){' (C)' if joueur.get('Capitaine') else ''}"
                color = "🟢"
            else:
                label = f"Ajouter {poste}{i+1}"
                color = "⚪"
            if cols[i].button(f"{color} {label}", key=f"{terrain_key}_{poste}_{i}"):
                st.session_state[f"edit_{terrain_key}"] = (poste, i)

    st.markdown("**Gardien**")
    poste_buttons("G", FORMATION[formation]["G"])
    st.markdown("**Défenseurs**")
    poste_buttons("D", FORMATION[formation]["D"])
    st.markdown("**Milieux**")
    poste_buttons("M", FORMATION[formation]["M"])
    st.markdown("**Attaquants**")
    poste_buttons("A", FORMATION[formation]["A"])

    # Formulaire sur clic
    edit_key = f"edit_{terrain_key}"
    if edit_key in st.session_state:
        poste, idx = st.session_state[edit_key]
        st.markdown(f"---\n**Ajouter/modifier {poste}{idx+1}**")
        # --- TOUS les joueurs proposés ---
        options = st.session_state.players["Nom"].tolist()
        choix = st.selectbox("Choisir un joueur", [""] + options)
        numero = st.number_input("Numéro de maillot", min_value=1, max_value=99, value=10, key=f"num_{terrain_key}_{poste}_{idx}")
        capitaine = st.checkbox("Capitaine", key=f"cap_{terrain_key}_{poste}_{idx}")
        if st.button("Valider ce joueur", key=f"valider_{terrain_key}_{poste}_{idx}"):
            if choix:
                terrain[poste][idx] = {
                    "Nom": choix,
                    "Numero": numero,
                    "Capitaine": capitaine
                }
                del st.session_state[edit_key]
                st.success("Joueur ajouté à la position !")
        if st.button("Retirer ce joueur", key=f"retirer_{terrain_key}_{poste}_{idx}"):
            terrain[poste][idx] = None
            del st.session_state[edit_key]
            st.info("Position libérée.")

    # Affichage récap
    st.markdown("**Composition actuelle :**")
    for poste in POSTES_ORDER:
        joueurs = [
            f"{j['Nom']} (#{j['Numero']}){' (C)' if j.get('Capitaine') else ''}"
            for j in terrain.get(poste, []) if j
        ]
        st.write(f"**{poste}** : {', '.join(joueurs) if joueurs else 'Aucun'}")

    # Sauvegarde temporaire en session
    st.session_state[terrain_key] = terrain
    st.session_state[f"formation_{terrain_key}"] = formation
    return terrain

# --- MENU PRINCIPAL ---
st.sidebar.title("⚽ Gestion Équipe AFC")
menu = st.sidebar.radio(
    "Menu",
    ["Database", "Créer Composition", "Mes Compos", "Matchs"]
)

# --- DATABASE (édition inline) ---
if menu == "Database":
    st.title("Base de données joueurs (édition directe)")
    st.markdown("Vous pouvez **éditer, supprimer ou ajouter** des joueurs directement dans le tableau ci-dessous. Les modifications sont enregistrées automatiquement.")
    edited_df = st.data_editor(
        st.session_state.players,
        num_rows="dynamic",
        use_container_width=True,
        key="data_edit"
    )
    # Validation et sauvegarde
    if st.button("Sauvegarder les modifications"):
        # On nettoie les lignes vides (sans nom)
        edited_df = edited_df.fillna("")
        edited_df = edited_df[edited_df["Nom"].str.strip() != ""]
        st.session_state.players = edited_df[PLAYER_COLS]
        save_players()
        reload_players()
        st.success("Base de joueurs mise à jour !")
    st.caption("Pour supprimer une ligne, videz le nom du joueur puis cliquez sur Sauvegarder.")

# --- CRÉER COMPOSITION ---
elif menu == "Créer Composition":
    st.title("Créer une nouvelle composition")
    nom_compo = st.text_input("Nom de la composition")
    formation = st.selectbox(
        "Formation", list(FORMATION.keys()),
        index=list(FORMATION.keys()).index(st.session_state.get("formation_create_compo", DEFAULT_FORMATION))
    )
    st.session_state["formation_create_compo"] = formation
    terrain = terrain_interactif(formation, "terrain_create_compo")

    if st.button("Sauvegarder la composition"):
        if not nom_compo.strip():
            st.warning("Veuillez donner un nom à la composition.")
        else:
            lineup = {
                "formation": formation,
                "details": terrain
            }
            st.session_state.lineups[nom_compo] = lineup
            save_lineups()
            st.success("Composition sauvegardée !")

# --- MES COMPOS ---
elif menu == "Mes Compos":
    st.title("Mes compositions sauvegardées")
    if not st.session_state.lineups:
        st.info("Aucune composition enregistrée.")
    else:
        for nom, compo in st.session_state.lineups.items():
            with st.expander(f"{nom} – {compo['formation']}"):
                for poste in POSTES_ORDER:
                    joueurs = [
                        f"{j['Nom']} (#{j['Numero']}){' (C)' if j.get('Capitaine') else ''}"
                        for j in compo['details'].get(poste, []) if j
                    ]
                    st.write(f"**{poste}** : {', '.join(joueurs) if joueurs else 'Aucun'}")
                if st.button(f"Supprimer {nom}", key=f"suppr_{nom}"):
                    del st.session_state.lineups[nom]
                    save_lineups()
                    st.experimental_rerun()

# --- MATCHS ---
elif menu == "Matchs":
    st.title("Gestion des matchs")
    tab1, tab2 = st.tabs(["Créer un match", "Mes matchs"])

    # Création d'un match
    with tab1:
        st.subheader("Créer un nouveau match")
        type_match = st.selectbox("Type de match", ["Championnat", "Coupe"])
        adversaire = st.text_input("Nom de l'adversaire")
        date = st.date_input("Date du match", value=datetime.today())
        heure = st.time_input("Heure du match")
        lieu = st.text_input("Lieu")
        use_compo = st.checkbox("Utiliser une composition enregistrée ?")
        if use_compo and st.session_state.lineups:
            compo_choice = st.selectbox("Choisir la composition", list(st.session_state.lineups.keys()))
            compo_data = st.session_state.lineups[compo_choice]
            formation = compo_data["formation"]
            import copy
            terrain = copy.deepcopy(compo_data["details"])
            st.session_state["formation_new_match"] = formation
            st.session_state["terrain_new_match"] = terrain
        else:
            formation = st.selectbox("Formation", list(FORMATION.keys()), key="match_formation")
            st.session_state["formation_new_match"] = formation
            terrain = terrain_interactif(formation, "terrain_new_match")

        remplaçants = st.multiselect("Remplaçants", st.session_state.players["Nom"].tolist())

        if st.button("Enregistrer le match"):
            match_id = f"{str(date)}_{adversaire}_{str(heure)}"
            st.session_state.matches[match_id] = {
                "type": type_match,
                "adversaire": adversaire,
                "date": str(date),
                "heure": str(heure),
                "lieu": lieu,
                "formation": formation,
                "details": st.session_state.get("terrain_new_match", terrain),
                "remplacants": remplaçants,
                "events": {},
                "score": "",
                "noted": False
            }
            save_matches()
            st.success("Match enregistré !")

    # Consultation/édition des matchs
    with tab2:
        if not st.session_state.matches:
            st.info("Aucun match enregistré.")
        else:
            for mid, match in st.session_state.matches.items():
                with st.expander(f"{match['date']} {match['heure']} vs {match['adversaire']} ({match['type']})"):
                    st.write(f"**Lieu :** {match['lieu']}")
                    st.write(f"**Formation :** {match['formation']}")
                    # Terrain interactif sur le match (édition possible tant que non noté)
                    if not match.get("noted", False):
                        st.session_state[f"formation_terrain_match_{mid}"] = match["formation"]
                        st.session_state[f"terrain_match_{mid}"] = match["details"]
                        terrain = terrain_interactif(match["formation"], f"terrain_match_{mid}")
                        if st.button("Mettre à jour la compo", key=f"maj_compo_{mid}"):
                            match["details"] = st.session_state.get(f"terrain_match_{mid}", match["details"])
                            save_matches()
                            st.success("Composition du match mise à jour.")
                    else:
                        for poste in POSTES_ORDER:
                            joueurs = [
                                f"{j['Nom']} (#{j['Numero']}){' (C)' if j.get('Capitaine') else ''}"
                                for j in match["details"].get(poste, []) if j
                            ]
                            st.write(f"**{poste}** : {', '.join(joueurs) if joueurs else 'Aucun'}")
                    st.write("**Remplaçants :** " + ", ".join(match.get("remplacants", [])))

                    # Statut match terminé
                    match_ended = st.checkbox("Match terminé", value=match.get("noted", False), key=f"ended_{mid}")
                    if match_ended and not match.get("noted", False):
                        st.write("### Saisie des stats du match")
                        joueurs_all = [j['Nom'] for p in POSTES_ORDER for j in match["details"].get(p, []) if j]
                        score = st.text_input("Score (ex: 2-1)", key=f"score_{mid}")
                        buteurs = st.multiselect("Buteurs", joueurs_all, key=f"buteurs_{mid}")
                        passeurs = st.multiselect("Passeurs", joueurs_all, key=f"passeurs_{mid}")
                        cj = st.multiselect("Cartons jaunes", joueurs_all, key=f"cj_{mid}")
                        cr = st.multiselect("Cartons rouges", joueurs_all, key=f"cr_{mid}")
                        notes = {}
                        for nom in joueurs_all:
                            notes[nom] = st.number_input(
                                f"Note pour {nom}", min_value=0.0, max_value=10.0, value=6.0, step=0.1, key=f"note_{mid}_{nom}"
                            )
                        if st.button("Valider le match", key=f"valide_{mid}"):
                            # Mise à jour stats joueurs dans la base
                            df = st.session_state.players
                            for nom in joueurs_all:
                                idx = df[df["Nom"] == nom].index
                                if len(idx):
                                    i = idx[0]
                                    df.at[i, "Sélections"] = df.at[i, "Sélections"] + 1
                                    df.at[i, "Titularisations"] = df.at[i, "Titularisations"] + 1
                                    if df.at[i, "Note générale"] > 0:
                                        df.at[i, "Note générale"] = round((df.at[i, "Note générale"] + notes[nom]) / 2, 2)
                                    else:
                                        df.at[i, "Note générale"] = notes[nom]
                            for nom in match.get("remplacants", []):
                                idx = df[df["Nom"] == nom].index
                                if len(idx):
                                    i = idx[0]
                                    df.at[i, "Sélections"] = df.at[i, "Sélections"] + 1
                            for nom in buteurs:
                                idx = df[df["Nom"] == nom].index
                                if len(idx):
                                    df.at[idx[0], "Buts"] = df.at[idx[0], "Buts"] + 1
                            for nom in passeurs:
                                idx = df[df["Nom"] == nom].index
                                if len(idx):
                                    df.at[idx[0], "Passes décisives"] = df.at[idx[0], "Passes décisives"] + 1
                            for nom in cj:
                                idx = df[df["Nom"] == nom].index
                                if len(idx):
                                    df.at[idx[0], "Cartons jaunes"] = df.at[idx[0], "Cartons jaunes"] + 1
                            for nom in cr:
                                idx = df[df["Nom"] == nom].index
                                if len(idx):
                                    df.at[idx[0], "Cartons rouges"] = df.at[idx[0], "Cartons rouges"] + 1
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
                    elif match.get("noted", False):
                        st.success("Match terminé")
                        st.write(f"**Score :** {match['score']}")
                        ev = match.get("events", {})
                        st.write("**Buteurs :** " + ", ".join(ev.get("buteurs", [])))
                        st.write("**Passeurs :** " + ", ".join(ev.get("passeurs", [])))
                        st.write("**Cartons jaunes :** " + ", ".join(ev.get("cartons_jaunes", [])))
                        st.write("**Cartons rouges :** " + ", ".join(ev.get("cartons_rouges", [])))
                        st.write("**Notes :**")
                        for nom, note in ev.get("notes", {}).items():
                            st.write(f"{nom} : {note}")

                    if st.button(f"Supprimer ce match", key=f"suppr_match_{mid}"):
                        del st.session_state.matches[mid]
                        save_matches()
                        st.experimental_rerun()
