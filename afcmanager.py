import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

# --- Fichiers de base de données ---
DB_FILE = "players_db.csv"
LINEUPS_FILE = "lineups.json"
MATCHES_FILE = "matches.json"

PLAYER_COLS = [
    "Nom", "Poste", "Club", "Titulaire", "Infos",
    "Buts", "Passes décisives", "Cartons jaunes", "Cartons rouges",
    "Sélections", "Titularisations", "Note générale"
]

FORMATION = {
    "4-4-2": {"G": 1, "D": 4, "M": 4, "A": 2},
    "4-3-3": {"G": 1, "D": 4, "M": 3, "A": 3},
    "3-5-2": {"G": 1, "D": 3, "M": 5, "A": 2},
    "3-4-3": {"G": 1, "D": 3, "M": 4, "A": 3},
    "5-3-2": {"G": 1, "D": 5, "M": 3, "A": 2},
}
DEFAULT_FORMATION = "4-4-2"

# --- Fonctions de base de données ---
def reload_players():
    if os.path.exists(DB_FILE):
        df = pd.read_csv(DB_FILE)
        for col in PLAYER_COLS:
            if col not in df.columns:
                df[col] = 0 if col not in ["Nom", "Poste", "Club", "Titulaire", "Infos"] else ""
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

# --- Initialisation Session ---
if "players" not in st.session_state:
    reload_players()
if "lineups" not in st.session_state:
    reload_lineups()
if "matches" not in st.session_state:
    reload_matches()
if "formation" not in st.session_state:
    st.session_state.formation = DEFAULT_FORMATION
if "terrain" not in st.session_state:
    st.session_state.terrain = {
        poste: [None for _ in range(nb)] for poste, nb in FORMATION[st.session_state.formation].items()
    }

# --- Affichage du menu principal ---
st.sidebar.title("⚽ Gestion Équipe AFC")
menu = st.sidebar.radio(
    "Menu",
    ["Terrain interactif", "Créer Composition", "Mes Compos", "Matchs", "Base Joueurs"]
)

# --- Terrain interactif ---
if menu == "Terrain interactif":
    st.header(f"Terrain interactif – {st.session_state.formation}")
    st.sidebar.subheader("Formation")
    formation = st.sidebar.selectbox(
        "Choix de la formation", list(FORMATION.keys()),
        index=list(FORMATION.keys()).index(st.session_state.formation)
    )
    if formation != st.session_state.formation:
        st.session_state.formation = formation
        st.session_state.terrain = {
            poste: [None for _ in range(nb)] for poste, nb in FORMATION[formation].items()
        }
        if "edit_poste" in st.session_state:
            del st.session_state["edit_poste"]

    st.write("Cliquez sur une case pour ajouter/modifier un joueur à cette position.")

    def poste_buttons(poste, n):
        cols = st.columns(n)
        for i in range(n):
            joueur = st.session_state.terrain[poste][i]
            if joueur:
                label = f"{joueur['Nom']} (#{joueur['Numero']}){' (C)' if joueur['Capitaine'] else ''}"
                color = "🟢"
            else:
                label = f"Ajouter {poste}{i+1}"
                color = "⚪"
            if cols[i].button(f"{color} {label}", key=f"{poste}_{i}"):
                st.session_state["edit_poste"] = (poste, i)

    st.markdown("#### Gardien")
    poste_buttons("G", FORMATION[st.session_state.formation]["G"])
    st.markdown("#### Défenseurs")
    poste_buttons("D", FORMATION[st.session_state.formation]["D"])
    st.markdown("#### Milieux")
    poste_buttons("M", FORMATION[st.session_state.formation]["M"])
    st.markdown("#### Attaquants")
    poste_buttons("A", FORMATION[st.session_state.formation]["A"])

    # Formulaire sur clic
    if "edit_poste" in st.session_state:
        poste, idx = st.session_state["edit_poste"]
        st.markdown(f"---\n### Ajouter/modifier {poste}{idx+1}")
        options = st.session_state.players[st.session_state.players["Poste"] == poste]["Nom"].tolist()
        choix = st.selectbox("Choisir un joueur", [""] + options)
        numero = st.number_input("Numéro de maillot", min_value=1, max_value=99, value=10)
        capitaine = st.checkbox("Capitaine")
        if st.button("Valider ce joueur"):
            if choix:
                st.session_state.terrain[poste][idx] = {
                    "Nom": choix,
                    "Numero": numero,
                    "Capitaine": capitaine
                }
                del st.session_state["edit_poste"]
                st.success("Joueur ajouté à la position !")
        if st.button("Retirer ce joueur", key=f"ret_{poste}_{idx}"):
            st.session_state.terrain[poste][idx] = None
            del st.session_state["edit_poste"]
            st.info("Position libérée.")

    st.markdown("---")
    st.markdown("#### Composition actuelle :")
    for poste in ["G", "D", "M", "A"]:
        joueurs = [
            f"{j['Nom']} (#{j['Numero']}){' (C)' if j['Capitaine'] else ''}"
            for j in st.session_state.terrain.get(poste, []) if j
        ]
        st.write(f"**{poste}** : {', '.join(joueurs) if joueurs else 'Aucun'}")

# --- Création de composition ---
elif menu == "Créer Composition":
    st.header("Créer une nouvelle composition")
    nom_compo = st.text_input("Nom de la composition")
    formation = st.selectbox(
        "Formation", list(FORMATION.keys()),
        index=list(FORMATION.keys()).index(st.session_state.formation)
    )
    if formation != st.session_state.formation:
        st.session_state.formation = formation
        st.session_state.terrain = {
            poste: [None for _ in range(nb)] for poste, nb in FORMATION[formation].items()
        }
        if "edit_poste" in st.session_state:
            del st.session_state["edit_poste"]

    st.subheader("Sélection des joueurs (voir menu Terrain interactif)")
    st.write("Utilisez le menu Terrain interactif pour sélectionner vos joueurs.")

    if st.button("Sauvegarder la composition"):
        if not nom_compo.strip():
            st.warning("Veuillez donner un nom à la composition.")
        else:
            lineup = {
                "formation": st.session_state.formation,
                "details": st.session_state.terrain
            }
            st.session_state.lineups[nom_compo] = lineup
            save_lineups()
            st.success("Composition sauvegardée !")

# --- Mes compositions sauvegardées ---
elif menu == "Mes Compos":
    st.header("Mes compositions sauvegardées")
    if not st.session_state.lineups:
        st.info("Aucune composition enregistrée.")
    else:
        for nom, compo in st.session_state.lineups.items():
            with st.expander(f"{nom} – {compo['formation']}"):
                for poste in ["G", "D", "M", "A"]:
                    joueurs = [
                        f"{j['Nom']} (#{j['Numero']}){' (C)' if j['Capitaine'] else ''}"
                        for j in compo['details'].get(poste, []) if j
                    ]
                    st.write(f"**{poste}** : {', '.join(joueurs) if joueurs else 'Aucun'}")
                if st.button(f"Supprimer {nom}", key=f"suppr_{nom}"):
                    del st.session_state.lineups[nom]
                    save_lineups()
                    st.experimental_rerun()

# --- Gestion des matchs ---
elif menu == "Matchs":
    st.header("Gestion des matchs")
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
            lineup_details = compo_data["details"]
        else:
            formation = st.selectbox("Formation", list(FORMATION.keys()), key="match_formation")
            lineup_details = {poste: [None for _ in range(nb)] for poste, nb in FORMATION[formation].items()}
            # Sélection rapide (pour compléter si besoin)
            st.write("Sélectionnez vos joueurs dans le menu Terrain interactif puis sauvegardez la compo.")

        # Remplaçants
        subs = st.multiselect("Remplaçants", st.session_state.players["Nom"].tolist())
        if st.button("Enregistrer le match"):
            match_id = f"{str(date)}_{adversaire}_{str(heure)}"
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

    # Consultation/édition des matchs
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
                        joueurs = [
                            f"{j['Nom']} (#{j['Numero']}){' (C)' if j['Capitaine'] else ''}"
                            for j in match["details"].get(poste, []) if j
                        ]
                        st.write(f"**{poste}** : {', '.join(joueurs) if joueurs else 'Aucun'}")
                    st.write("**Remplaçants :** " + ", ".join(match.get("remplacants", [])))

                    # Saisie post match
                    if not match.get("noted", False):
                        st.write("### Saisie des stats du match")
                        score = st.text_input("Score (ex: 2-1)", key=f"score_{mid}")
                        joueurs_all = [j['Nom'] for p in ["G", "D", "M", "A"] for j in match["details"].get(p, []) if j]
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
                                    # Moyenne pondérée des notes
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

                    if st.button(f"Supprimer ce match", key=f"suppr_match_{mid}"):
                        del st.session_state.matches[mid]
                        save_matches()
                        st.experimental_rerun()

# --- Base de données joueurs ---
elif menu == "Base Joueurs":
    st.header("Base de données des joueurs")
    st.write("### Joueurs enregistrés")
    st.dataframe(st.session_state.players)

    st.markdown("---")
    st.write("### Ajouter un joueur à la base")
    with st.form("add_player"):
        nom = st.text_input("Nom")
        poste = st.selectbox("Poste", ["G", "D", "M", "A"])
        club = st.text_input("Club")
        titulaire = st.checkbox("Titulaire probable", value=True)
        infos = st.text_input("Infos complémentaires")
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
                pd.DataFrame([new_row], columns=PLAYER_COLS)
            ], ignore_index=True)
            save_players()
            reload_players()
            st.success(f"{nom} ajouté à la base.")
