import streamlit as st

FORMATION = {
    "4-4-2": [
        ("Gardien", 50, 90),
        ("Arrière Gauche", 15, 70),
        ("Défenseur Central Gauche", 35, 70),
        ("Défenseur Central Droit", 65, 70),
        ("Arrière Droit", 85, 70),
        ("Milieu Gauche", 15, 45),
        ("Milieu Central Gauche", 35, 45),
        ("Milieu Central Droit", 65, 45),
        ("Milieu Droit", 85, 45),
        ("Attaquant Gauche", 30, 20),
        ("Attaquant Droit", 70, 20),
    ],
    "4-3-3": [
        ("Gardien", 50, 90),
        ("Arrière Gauche", 15, 70),
        ("Défenseur Central Gauche", 35, 70),
        ("Défenseur Central Droit", 65, 70),
        ("Arrière Droit", 85, 70),
        ("Milieu Gauche", 25, 45),
        ("Milieu Central", 50, 45),
        ("Milieu Droit", 75, 45),
        ("Attaquant Gauche", 20, 20),
        ("Attaquant Central", 50, 20),
        ("Attaquant Droit", 80, 20),
    ],
    "3-5-2": [
        ("Gardien", 50, 90),
        ("Défenseur Gauche", 25, 70),
        ("Défenseur Central", 50, 70),
        ("Défenseur Droit", 75, 70),
        ("Milieu Gauche", 10, 45),
        ("Milieu Central Gauche", 30, 45),
        ("Milieu Central", 50, 45),
        ("Milieu Central Droit", 70, 45),
        ("Milieu Droit", 90, 45),
        ("Attaquant Gauche", 30, 20),
        ("Attaquant Droit", 70, 20),
    ],
}


def choix_joueurs_interface(
    formation, joueurs, data, key_prefix, nombre_titulaires=11, nombre_remplacants=7
):
    """Affiche une interface pour choisir les joueurs titulaires et remplaçants."""
    st.subheader("Titulaires")
    titulaires = []
    postes = FORMATION[formation]
    for i, (poste, _, _) in enumerate(postes):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.write(f"**{poste}**")
        with col2:
            options = [""] + [
                j["Nom"] for j in joueurs
            ]  # Adds an empty string for no selection
            selected_name = st.selectbox(
                f"Joueur {i+1}", options, key=f"{key_prefix}_titulaire_{i}"
            )
            joueur = next((j for j in joueurs if j["Nom"] == selected_name), None)
            titulaires.append(joueur)
        with col3:
            numero = st.number_input(
                "Numéro",
                min_value=1,
                max_value=99,
                value=i + 1,
                key=f"{key_prefix}_numero_{i}",
            )
            if joueur:
                joueur["Numero"] = numero

    st.subheader("Remplaçants")
    remplacants = []
    for i in range(nombre_remplacants):
        col1, col2 = st.columns([1, 2])
        with col1:
            st.write(f"Remplaçant {i+1}")
        with col2:
            options = [""] + [j["Nom"] for j in joueurs]
            selected_name = st.selectbox(
                f"Remplaçant {i+1}",
                options,
                key=f"{key_prefix}_remplacant_{i}",
            )
            joueur = next((j for j in joueurs if j["Nom"] == selected_name), None)
            remplacants.append(joueur)
    return titulaires, remplacants


def terrain_viz_simple(formation, titulaires, remplacants, captain_name):
    titulaires = titulaires or []
    remplacants = remplacants or []
    postes = FORMATION[formation]

    # Define CSS styles
    st.markdown(
        """
        <style>
        .field {
            background: linear-gradient(180deg, #4db367 0%, #245c32 100%);
            border-radius: 30px;
            border: 3px solid #fff;
            overflow: hidden;
            position: relative;
            width: 100%;
            max-width: 480px;
            aspect-ratio: 2/3;
            margin: auto;
        }
        .player {
            position: absolute;
            width: 13%;
            min-width: 50px;
            text-align: center;
            transform: translate(-50%, -50%);
        }
        .player-circle {
            background: #1976D2;
            color: #fff;
            width: 3.5em;
            height: 3.5em;
            border-radius: 10px;
            border: 3px solid #fff;
            border: 3px solid #fff;
            display: flex;
            align-items: center;
            justify-content: center;
            font-size: 1em;
            font-weight: bold;
            position: relative;
            margin: auto;
        }
        .captain-indicator {
            position: absolute;
            top: -13px;
            right: -12px;
            background: #FFD700;
            color: #000;
            padding: 2px 6px;
            border-radius: 10px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .player-name {
            font-size: 0.95em;
            color: #fff;
            text-shadow: 0 1px 2px #000a;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Create the field container
    field_html = '<div class="field">'
    idx = 0

    for poste, x, y in postes:
        joueur = titulaires[idx] if idx < len(titulaires) else None
        if joueur and isinstance(joueur, dict) and joueur.get("Nom"):
            is_cap = joueur.get("Nom") == captain_name

            field_html += f"""
            <div class="player" style="left:{x}%; top:{y}%;">
                <div class="player-circle" style="border-color:{'#FFD700' if is_cap else '#fff'}">
                    {joueur.get("Numero", "")}
                    {'<span class="captain-indicator">C</span>' if is_cap else ''}
                </div>
                <div class="player-name">{joueur.get("Nom")}</div>
            </div>
            """
        idx += 1

    field_html += "</div>"

    st.markdown(field_html, unsafe_allow_html=True)

    remp_aff = [f'{r.get("Nom")} (#{r.get("Numero")})' for r in remplacants if isinstance(r, dict) and r.get("Nom")]
    if remp_aff:
        st.markdown("**Remplaçants** : " + ", ".join(remp_aff))


def tab_compositions(data):
    st.header("Compositions d'Équipe")
    with st.expander("Gestion des Compositions"):
        st.subheader("Nouvelle Composition")
        nom_composition = st.text_input("Nom de la Composition")
        formation = st.selectbox("Formation", list(FORMATION.keys()))
        joueurs = data["joueurs"] if "joueurs" in data else []

        if joueurs:
            titulaires, remplacants = choix_joueurs_interface(
                formation, joueurs, data, "composition"
            )

            captain_name = st.selectbox(
                "Capitaine",
                [""] + [j["Nom"] for j in titulaires if j],
                key="captain_composition",
            )

            if st.button("Enregistrer la Composition"):
                if nom_composition and formation and titulaires:
                    data["compositions"] = data.get("compositions", []) + [
                        {
                            "Nom": nom_composition,
                            "Formation": formation,
                            "Titulaires": titulaires,
                            "Remplacants": remplacants,
                            "Capitaine": captain_name,
                        }
                    ]
                    st.success("Composition enregistrée!")
                else:
                    st.error(
                        "Veuillez remplir tous les champs et sélectionner les joueurs titulaires."
                    )
        else:
            st.warning("Ajoutez des joueurs dans l'onglet 'Joueurs' d'abord.")

    with st.expander("Visualisation des Compositions"):
        if "compositions" in data and data["compositions"]:
            composition_names = [c["Nom"] for c in data["compositions"]]
            selected_composition_name = st.selectbox(
                "Sélectionner une Composition à visualiser", composition_names
            )
            selected_composition = next(
                (c for c in data["compositions"] if c["Nom"] == selected_composition_name),
                None,
            )

            if selected_composition:
                st.subheader(f"Composition: {selected_composition['Nom']}")
                terrain_viz_simple(
                    selected_composition["Formation"],
                    selected_composition["Titulaires"],
                    selected_composition["Remplacants"],
                    selected_composition["Capitaine"],
                )
            else:
                st.write("Aucune composition sélectionnée.")
        else:
            st.info("Aucune composition enregistrée.")


def tab_matchs(data):
    st.header("Gestion des Matchs")
    with st.expander("Planification des Matchs"):
        st.subheader("Nouveau Match")
        date_match = st.date_input("Date du Match")
        adversaire = st.text_input("Adversaire")
        lieu = st.text_input("Lieu du Match")
        compositions = data.get("compositions", [])
        composition_options = [""] + [c["Nom"] for c in compositions]
        selected_composition_name = st.selectbox(
            "Composition", composition_options, key="composition_match"
        )
        selected_composition = next(
            (c for c in compositions if c["Nom"] == selected_composition_name), None
        )

        if st.button("Planifier le Match"):
            if date_match and adversaire and lieu and selected_composition:
                data["matchs"] = data.get("matchs", []) + [
                    {
                        "Date": date_match,
                        "Adversaire": adversaire,
                        "Lieu": lieu,
                        "Composition": selected_composition_name,
                    }
                ]
                st.success("Match planifié!")
            else:
                st.error("Veuillez remplir tous les champs.")

    with st.expander("Visualisation des Matchs"):
        if "matchs" in data and data["matchs"]:
            match_dates = [m["Date"] for m in data["matchs"]]
            selected_date = st.selectbox(
                "Sélectionner un Match par Date", match_dates, key="match_date"
            )
            selected_match = next(
                (m for m in data["matchs"] if m["Date"] == selected_date), None
            )

            if selected_match:
                st.subheader(
                    f"Match du {selected_match['Date']} contre {selected_match['Adversaire']} à {selected_match['Lieu']}"
                )
                selected_composition = next(
                    (
                        c
                        for c in data["compositions"]
                        if c["Nom"] == selected_match["Composition"]
                    ),
                    None,
                )
            if selected_composition:
                st.subheader(f"Composition: {selected_composition['Nom']}")
                terrain_viz_simple(
                    selected_composition["Formation"],
                    selected_composition["Titulaires"],
                    selected_composition["Remplacants"],
                    selected_composition["Capitaine"],
                )
            else:
                st.write("Aucune composition sélectionnée.")
        else:
            st.info("Aucun match planifié.")


def tab_joueurs(data):
    st.header("Gestion des Joueurs")
    with st.expander("Ajouter un Joueur"):
        nouveau_nom = st.text_input("Nom du Joueur")
        nouveau_poste = st.text_input("Poste du Joueur")
        if st.button("Ajouter"):
            if nouveau_nom:
                data["joueurs"] = data.get("joueurs", []) + [
                    {"Nom": nouveau_nom, "Poste": nouveau_poste}
                ]
                st.success(f"Joueur {nouveau_nom} ajouté!")
            else:
                st.error("Veuillez entrer le nom du joueur.")

    with st.expander("Liste des Joueurs"):
        if "joueurs" in data and data["joueurs"]:
            st.dataframe(data["joueurs"])
        else:
            st.info("Aucun joueur enregistré.")


def main():
    st.title("AFC Manager")
    # Initialize the session state
    if "data" not in st.session_state:
        st.session_state["data"] = {}

    # Sidebar for navigation
    tabs = ["Joueurs", "Compositions", "Matchs"]
    selected_tab = st.sidebar.selectbox("Choisir l'onglet", tabs)

    # Display the selected tab
    if selected_tab == "Joueurs":
        tab_joueurs(st.session_state["data"])
    elif selected_tab == "Compositions":
        tab_compositions(st.session_state["data"])
    elif selected_tab == "Matchs":
        tab_matchs(st.session_state["data"])


if __name__ == "__main__":
    main()