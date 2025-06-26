import streamlit as st
import pandas as pd
import os

DB_FILE = "players_db.csv"
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

# Initialisation
if "players" not in st.session_state:
    reload_players()

if "formation" not in st.session_state:
    st.session_state.formation = DEFAULT_FORMATION

if "terrain" not in st.session_state:
    # Un dict {poste: [joueur ou None]}
    st.session_state.terrain = {
        poste: [None for _ in range(nb)] for poste, nb in FORMATION[st.session_state.formation].items()
    }

# -- Affichage du titre et du menu
st.title("AFC Manager – Terrain interactif")
menu = st.sidebar.radio("Menu", ["Terrain interactif", "Base Joueurs"])

# -- Changement de formation
if menu == "Terrain interactif":
    st.sidebar.subheader("Formation")
    formation = st.sidebar.selectbox("Choix de la formation", list(FORMATION.keys()), index=list(FORMATION.keys()).index(st.session_state.formation))
    if formation != st.session_state.formation:
        st.session_state.formation = formation
        st.session_state.terrain = {
            poste: [None for _ in range(nb)] for poste, nb in FORMATION[formation].items()
        }
        if "edit_poste" in st.session_state:
            del st.session_state["edit_poste"]

    st.header(f"Terrain interactif – {st.session_state.formation}")
    st.write("Cliquez sur une case pour ajouter/modifier un joueur à cette position.")

    # Affichage du terrain par lignes
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

    # Formulaire dynamique si un poste est cliqué
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

if menu == "Base Joueurs":
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
