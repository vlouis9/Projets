import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

DATA_FILE = "afcdata.json"
PLAYER_COLS = ["Nom", "Poste", "Infos"]
PLAYER_DEFAULTS = {"Nom": "", "Poste": "G", "Infos": ""}

FORMATION = {
    "4-4-2": {"G": 1, "D": 4, "M": 4, "A": 2},
    "4-3-3": {"G": 1, "D": 4, "M": 3, "A": 3},
    "3-5-2": {"G": 1, "D": 3, "M": 5, "A": 2},
    "3-4-3": {"G": 1, "D": 3, "M": 4, "A": 3},
    "5-3-2": {"G": 1, "D": 5, "M": 3, "A": 2},
}
POSTES_ORDER = ["G", "D", "M", "A"]
DEFAULT_FORMATION = "4-4-2"
MAX_REMPLACANTS = 5

# Normalized positions for each formation for visual field
FORMATION_COORDS = {
    "4-4-2": {
        "G": [(0.07, 0.5)],
        "D": [(0.27, 0.12), (0.27, 0.32), (0.27, 0.68), (0.27, 0.88)],
        "M": [(0.54, 0.12), (0.54, 0.32), (0.54, 0.68), (0.54, 0.88)],
        "A": [(0.83, 0.35), (0.83, 0.65)],
    },
    "4-3-3": {
        "G": [(0.07, 0.5)],
        "D": [(0.27, 0.12), (0.27, 0.32), (0.27, 0.68), (0.27, 0.88)],
        "M": [(0.54, 0.22), (0.54, 0.5), (0.54, 0.78)],
        "A": [(0.83, 0.18), (0.83, 0.5), (0.83, 0.82)],
    },
    "3-5-2": {
        "G": [(0.07, 0.5)],
        "D": [(0.27, 0.2), (0.27, 0.5), (0.27, 0.8)],
        "M": [(0.47, 0.09), (0.47, 0.28), (0.54, 0.5), (0.47, 0.72), (0.47, 0.91)],
        "A": [(0.83, 0.35), (0.83, 0.65)],
    },
    "3-4-3": {
        "G": [(0.07, 0.5)],
        "D": [(0.27, 0.2), (0.27, 0.5), (0.27, 0.8)],
        "M": [(0.54, 0.2), (0.54, 0.4), (0.54, 0.6), (0.54, 0.8)],
        "A": [(0.83, 0.18), (0.83, 0.5), (0.83, 0.82)],
    },
    "5-3-2": {
        "G": [(0.07, 0.5)],
        "D": [(0.22, 0.08), (0.27, 0.25), (0.27, 0.5), (0.27, 0.75), (0.22, 0.92)],
        "M": [(0.54, 0.22), (0.54, 0.5), (0.54, 0.78)],
        "A": [(0.83, 0.35), (0.83, 0.65)],
    }
}

# --- Utilitaires persistance ---
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

# --- Calcul dynamique des stats joueurs ---
def compute_player_stats(joueur_nom):
    buts = passes = cj = cr = selections = titularisations = note_sum = note_count = hdm = 0
    for match in st.session_state.matches.values():
        details = match.get("details", {})
        joueurs = [j for p in POSTES_ORDER for j in details.get(p, []) if j and j["Nom"] == joueur_nom]
        is_titulaire = bool(joueurs)
        if is_titulaire or joueur_nom in match.get("remplacants", []):
            selections += 1
        if is_titulaire:
            titularisations += 1
        events = match.get("events", {})
        buts += events.get("buteurs", {}).get(joueur_nom, 0)
        passes += events.get("passeurs", {}).get(joueur_nom, 0)
        cj += events.get("cartons_jaunes", {}).get(joueur_nom, 0)
        cr += events.get("cartons_rouges", {}).get(joueur_nom, 0)
        if "notes" in events and joueur_nom in events["notes"]:
            note_sum += events["notes"][joueur_nom]
            note_count += 1
        if "homme_du_match" in match and match["homme_du_match"] == joueur_nom:
            hdm += 1
    note = round(note_sum / note_count, 2) if note_count else 0
    buts_passes = buts + passes
    decisif_par_match = round(buts_passes / selections, 2) if selections > 0 else 0
    return {
        "Buts": buts,
        "Passes décisives": passes,
        "Buts + Passes": buts_passes,
        "Décisif par match": decisif_par_match,
        "Cartons jaunes": cj,
        "Cartons rouges": cr,
        "Sélections": selections,
        "Titularisations": titularisations,
        "Note générale": note,
        "Homme du match": hdm
    }

# --- Initialisation session ---
if "players" not in st.session_state:
    reload_all()
if "lineups" not in st.session_state:
    reload_all()
if "matches" not in st.session_state:
    reload_all()
if "formation" not in st.session_state:
    st.session_state.formation = DEFAULT_FORMATION

# --- Fonctions de download/upload ---
def download_upload_buttons():
    st.markdown("### 📥 Sauvegarde/Import global (tout-en-un)")
    st.download_button(
        label="Télécharger toutes les données (JSON)",
        data=json.dumps({
            "players": st.session_state.players.to_dict(orient="records"),
            "lineups": st.session_state.lineups,
            "matches": st.session_state.matches,
        }, indent=2),
        file_name=DATA_FILE,
        mime="application/json"
    )
    up_json = st.file_uploader("Importer un fichier complet (JSON)", type="json", key="upload_all")
    if up_json:
        try:
            data = json.load(up_json)
            st.session_state.players = pd.DataFrame(data.get("players", []))
            st.session_state.lineups = data.get("lineups", {})
            st.session_state.matches = data.get("matches", {})
            st.success("Données importées !")
        except Exception as e:
            st.error(f"Erreur à l'import : {e}")

# --- Sélection Remplaçants Interactifs ---
def remplaçants_interactif(key, titulaires):
    if f"remp_{key}" not in st.session_state:
        st.session_state[f"remp_{key}"] = [None] * MAX_REMPLACANTS
    remps = st.session_state[f"remp_{key}"]

    dispo = [n for n in st.session_state.players["Nom"] if n not in titulaires and n not in remps if n]
    for i in range(MAX_REMPLACANTS):
        current = remps[i]
        options = dispo + ([current] if current and current not in dispo else [])
        choix = st.selectbox(
            f"Remplaçant {i+1}",
            [""] + options,
            index=(options.index(current)+1) if current in options else 0,
            key=f"remp_choice_{key}_{i}"
        )
        remps[i] = choix if choix else None
        dispo = [n for n in dispo if n != choix]
    st.session_state[f"remp_{key}"] = remps
    return [r for r in remps if r]

# --- Helper for initializing the terrain ---
def terrain_init(formation):
    return {poste: [None for _ in range(FORMATION[formation][poste])] for poste in POSTES_ORDER}

# --- Visual terrain interactif (NEW) ---
def terrain_interactif(formation, terrain_key):
    # Field dimensions for rendering
    field_width, field_height = 750, 500

    if terrain_key not in st.session_state or st.session_state.get(f"formation_{terrain_key}", None) != formation:
        st.session_state[terrain_key] = terrain_init(formation)
        st.session_state[f"formation_{terrain_key}"] = formation
    terrain = st.session_state[terrain_key]

    # Editing state
    edit_key = f"edit_{terrain_key}"
    if edit_key not in st.session_state:
        st.session_state[edit_key] = None

    # HTML for field and players
    html = f"""
    <div style="position:relative;width:{field_width}px;height:{field_height}px;background:#2d7d46;
                border-radius:30px;box-shadow:0 0 20px #333;border:5px solid #fff;overflow:hidden;">
        <!-- Center line -->
        <div style="position:absolute;left:{field_width//2-2}px;top:0;width:4px;height:{field_height}px;background:#fff;opacity:0.3;"></div>
        <!-- Circle -->
        <div style="position:absolute;left:{field_width//2-60}px;top:{field_height//2-60}px;width:120px;height:120px;border:4px solid #fff;border-radius:50%;opacity:0.3;"></div>
    """

    for poste, coords_list in FORMATION_COORDS[formation].items():
        for idx, (x, y) in enumerate(coords_list):
            joueur = terrain[poste][idx] if idx < len(terrain[poste]) else None
            left = int(x * field_width)
            top = int(y * field_height)
            base_style = (
                f"position:absolute;left:{left-36}px;top:{top-36}px;width:72px;height:72px;"
                "border-radius:50%;display:flex;flex-direction:column;align-items:center;"
                "justify-content:center;box-shadow:0 4px 18px #1118;"
                "cursor:pointer;transition:transform 0.1s;"
            )
            if joueur:
                color = "#fff"
                border = "#2d7d46"
                icon = f"""<b style="font-size:26px;color:#246;">{joueur.get('Numero','')}</b>"""
                name = f"""<span style="font-size:13px;font-weight:600;color:#222;">{joueur['Nom']}</span>"""
                cap = '<span style="position:absolute;top:4px;right:7px;color:gold;font-size:19px;">★</span>' if joueur.get("Capitaine") else ""
                player_html = (
                    f"""<div style="{base_style}background:{color};border:3px solid {border};position:absolute;"
                            onclick="window.parent.postMessage({{'edit':'{poste}_{idx}'}}, '*')"
                            title="Cliquer pour modifier">
                            {icon}{name}{cap}
                        </div>"""
                )
            else:
                color = "#d3e6d7"
                border = "#aaa"
                player_html = (
                    f"""<div style="{base_style}background:{color};border:3px dashed {border};color:#555;font-size:42px;position:absolute;"
                            onclick="window.parent.postMessage({{'edit':'{poste}_{idx}'}}, '*')"
                            title="Ajouter un joueur">
                            +
                        </div>"""
                )
            html += player_html

    html += "</div>"

    # Display field
    st.markdown(html, unsafe_allow_html=True)

    # JavaScript to catch player clicks (Streamlit workaround!)
    st.components.v1.html("""
    <script>
    window.addEventListener('message', (ev) => {
        if(ev.data.edit){
            const s = window.parent.document.querySelector('iframe[title^="streamlit"]');
            if(s) s.contentWindow.postMessage(ev.data, '*');
        }
    });
    window.addEventListener('message', (ev) => {
        if(ev.data.edit){
            window.parent.postMessage(ev.data, '*');
        }
    });
    </script>
    """, height=0)

    # Use Streamlit event handling for player edit
    edit = st.query_params().get("edit", [None])[0]
    if edit and edit_key in st.session_state:
        poste, idx = edit.split("_")
        idx = int(idx)
        st.session_state[edit_key] = (poste, idx)
        # Clean up URL param
        st.experimental_set_query_params()

    if st.session_state[edit_key]:
        poste, idx = st.session_state[edit_key]
        st.markdown(f"### Ajouter/Modifier {poste}{idx+1}")
        joueurs_sur_terrain = set(
            j["Nom"]
            for p in POSTES_ORDER
            for j in terrain.get(p, [])
            if j and isinstance(j, dict) and j.get("Nom")
        )
        joueur_courant = terrain[poste][idx]["Nom"] if terrain[poste][idx] else None
        if joueur_courant:
            joueurs_sur_terrain = joueurs_sur_terrain - {joueur_courant}
        all_options = st.session_state.players["Nom"].tolist()
        options = [n for n in all_options if n not in joueurs_sur_terrain]
        choix = st.selectbox("Choisir un joueur", [""] + options, key=f"choix_{terrain_key}_{poste}_{idx}")
        numero = st.number_input("Numéro de maillot", min_value=1, max_value=99, value=terrain[poste][idx]["Numero"] if terrain[poste][idx] else 10, key=f"num_{terrain_key}_{poste}_{idx}")
        capitaine = st.checkbox("Capitaine", value=terrain[poste][idx]["Capitaine"] if terrain[poste][idx] else False, key=f"cap_{terrain_key}_{poste}_{idx}")
        col1, col2 = st.columns(2)
        if col1.button("Valider ce joueur", key=f"valider_{terrain_key}_{poste}_{idx}"):
            if choix:
                terrain[poste][idx] = {
                    "Nom": choix,
                    "Numero": numero,
                    "Capitaine": capitaine
                }
                st.session_state[edit_key] = None
                st.session_state[terrain_key] = terrain
                st.experimental_rerun()
        if col2.button("Retirer ce joueur", key=f"retirer_{terrain_key}_{poste}_{idx}"):
            terrain[poste][idx] = None
            st.session_state[edit_key] = None
            st.session_state[terrain_key] = terrain
            st.experimental_rerun()

    with st.expander("Composition actuelle"):
        for poste in POSTES_ORDER:
            joueurs = [
                f"{j['Nom']} (#{j.get('Numero', '')}){' (C)' if j.get('Capitaine') else ''}"
                for j in terrain.get(poste, []) if j
            ]
            st.write(f"**{poste}** : {', '.join(joueurs) if joueurs else 'Aucun'}")
    st.session_state[terrain_key] = terrain
    st.session_state[f"formation_{terrain_key}"] = formation
    return terrain

# --- Visual remplaçants ---
def render_replacements(remplaçants, players_df):
    st.markdown("#### Remplaçants")
    cols = st.columns(max(len(remplaçants), 1))
    for idx, remp in enumerate(remplaçants):
        if remp:
            joueur = players_df.loc[players_df["Nom"] == remp]
            numero = int(joueur["Numero"].values[0]) if "Numero" in joueur and len(joueur["Numero"].values)>0 else ""
            nom = remp
            cols[idx].markdown(
                f"""
                <div style="width:60px;height:60px;border-radius:50%;background:#ddd;
                            border:2px solid #888;display:flex;flex-direction:column;
                            align-items:center;justify-content:center;">
                  <b>{numero}</b>
                  <span style="font-size:13px;">{nom}</span>
                </div>
                """, unsafe_allow_html=True
            )
        else:
            cols[idx].markdown(
                "<div style='width:60px;height:60px;border-radius:50%;background:#eee;"
                "border:2px dashed #bbb;display:flex;align-items:center;justify-content:center;'>+</div>",
                unsafe_allow_html=True
            )

# --- MENU PRINCIPAL ---
st.sidebar.title("⚽ Gestion Équipe AFC")
menu = st.sidebar.radio(
    "Menu",
    ["Database", "Compositions", "Matchs", "Sauvegarde / Import"]
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
    if st.button("Sauvegarder les modifications"):
        edited_df = edited_df.fillna("")
        edited_df = edited_df[edited_df["Nom"].str.strip() != ""]
        st.session_state.players = edited_df[PLAYER_COLS]
        save_all()
        reload_all()
        st.success("Base de joueurs mise à jour !")
    st.caption("Pour supprimer une ligne, videz le nom du joueur puis cliquez sur Sauvegarder.")

    # Affichage dynamique des stats
    st.markdown("### Statistiques dynamiques (calculées à partir des matchs présents)")
    stats_cols = [
        "Nom", "Poste", "Infos", "Buts", "Passes décisives", "Buts + Passes", "Décisif par match",
        "Cartons jaunes", "Cartons rouges", "Sélections", "Titularisations", "Note générale", "Homme du match"
    ]
    stats_data = []
    for _, row in st.session_state.players.iterrows():
        s = compute_player_stats(row["Nom"])
        stats_data.append({**row, **s})
    st.dataframe(pd.DataFrame(stats_data, columns=stats_cols))

# --- COMPOSITIONS ---
elif menu == "Compositions":
    st.title("Gestion des compositions")
    tab1, tab2 = st.tabs(["Créer une composition", "Mes compositions"])
    # Edition/Création
    with tab1:
        edit_key = "edit_compo"
        edit_compo = st.session_state.get(edit_key, None)
        if edit_compo:
            nom_compo, loaded = edit_compo
            st.info(f"Édition de la compo : {nom_compo}")
            st.session_state["formation_create_compo"] = loaded["formation"]
            st.session_state["terrain_create_compo"] = loaded["details"]
            del st.session_state[edit_key]
        nom_compo = st.text_input("Nom de la composition", value=nom_compo if edit_compo else "")
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
                save_all()
                st.success("Composition sauvegardée !")
        # Ajout visuel des remplaçants sous le terrain (optionnel ici)
        # render_replacements([], st.session_state.players)
    # Liste/Edition
    with tab2:
        if not st.session_state.lineups:
            st.info("Aucune composition enregistrée.")
        else:
            for nom, compo in st.session_state.lineups.items():
                with st.expander(f"{nom} – {compo['formation']}"):
                    for poste in POSTES_ORDER:
                        joueurs = [
                            f"{j['Nom']} (#{j.get('Numero', '')}){' (C)' if j.get('Capitaine') else ''}"
                            for j in compo['details'].get(poste, []) if j
                        ]
                        st.write(f"**{poste}** : {', '.join(joueurs) if joueurs else 'Aucun'}")
                    col1, col2 = st.columns(2)
                    if col1.button(f"Éditer {nom}", key=f"edit_{nom}"):
                        st.session_state["edit_compo"] = (nom, compo)
                        st.experimental_rerun()
                    if col2.button(f"Supprimer {nom}", key=f"suppr_{nom}"):
                        del st.session_state.lineups[nom]
                        save_all()
                        st.experimental_rerun()

# --- MATCHS ---
elif menu == "Matchs":
    st.title("Gestion des matchs")
    tab1, tab2 = st.tabs(["Créer un match", "Mes matchs"])

    with tab1:
        if st.button("Réinitialiser la création du match"):
            for k in [
                "terrain_new_match", "formation_new_match",
                "remp_new_match", "nom_match_sugg", "adversaire", "lieu"
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            st.experimental_rerun()

        type_match = st.selectbox("Type de match", ["Championnat", "Coupe"])
        adversaire = st.text_input("Nom de l'adversaire", key="adversaire")
        date = st.date_input("Date du match", value=datetime.today())
        heure = st.time_input("Heure du match")
        lieu = st.text_input("Lieu", key="lieu")
        # Suggestion automatique du nom
        nom_sugg = f"{date.strftime('%Y-%m-%d')} vs {adversaire}" if adversaire else f"{date.strftime('%Y-%m-%d')}"
        nom_match = st.text_input("Nom du match", value=st.session_state.get("nom_match_sugg", nom_sugg), key="nom_match_sugg")

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

        # Titulaires pour éviter doublons remplaçants
        tous_titulaires = [j["Nom"] for p in POSTES_ORDER for j in st.session_state.get("terrain_new_match", terrain).get(p, []) if j]
        remplaçants = remplaçants_interactif("new_match", tous_titulaires)
        render_replacements(remplaçants, st.session_state.players)

        # Enregistrer la compo depuis la création d'un match
        if st.button("Enregistrer cette compo"):
            name_compo = st.text_input("Nom pour la compo à enregistrer", value=nom_match)
            if name_compo:
                lineup = {
                    "formation": formation,
                    "details": st.session_state.get("terrain_new_match", terrain)
                }
                st.session_state.lineups[name_compo] = lineup
                save_all()
                st.success("Composition sauvegardée !")

        if st.button("Enregistrer le match"):
            match_id = nom_match
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
                "noted": False,
                "homme_du_match": ""
            }
            save_all()
            st.success("Match enregistré !")
            st.experimental_rerun()

    with tab2:
        if not st.session_state.matches:
            st.info("Aucun match enregistré.")
        else:
            for mid, match in st.session_state.matches.items():
                with st.expander(f"{match['date']} {match['heure']} vs {match['adversaire']} ({match['type']})"):
                    statut = "Terminé" if match.get("noted", False) else "En cours"
                    st.write(f"**Statut :** {statut}")
                    if match.get("noted", False):
                        st.success("Match terminé")
                        st.write(f"**Score :** {match.get('score','')}")
                        ev = match.get("events", {})
                        st.write("**Buteurs :**")
                        for nom, nb in ev.get("buteurs", {}).items():
                            st.write(f"- {nom} : {nb}")
                        st.write("**Passeurs :**")
                        for nom, nb in ev.get("passeurs", {}).items():
                            st.write(f"- {nom} : {nb}")
                        st.write("**Cartons jaunes :**")
                        for nom, nb in ev.get("cartons_jaunes", {}).items():
                            st.write(f"- {nom} : {nb}")
                        st.write("**Cartons rouges :**")
                        for nom, nb in ev.get("cartons_rouges", {}).items():
                            st.write(f"- {nom} : {nb}")
                        st.write(f"**Homme du match :** {match.get('homme_du_match','')}")
                    st.write(f"**Lieu :** {match['lieu']}")
                    st.write(f"**Formation :** {match['formation']}")
                    for poste in POSTES_ORDER:
                        joueurs = [
                            f"{j['Nom']} (#{j.get('Numero', '')}){' (C)' if j.get('Capitaine') else ''}"
                            for j in match["details"].get(poste, []) if j
                        ]
                        st.write(f"**{poste}** : {', '.join(joueurs) if joueurs else 'Aucun'}")
                    st.write("**Remplaçants :** " + ", ".join(match.get("remplacants", [])))
                    render_replacements(match.get("remplacants", []), st.session_state.players)

                    if not match.get("noted", False):
                        st.session_state[f"formation_terrain_match_{mid}"] = match["formation"]
                        st.session_state[f"terrain_match_{mid}"] = match["details"]
                        terrain = terrain_interactif(match["formation"], f"terrain_match_{mid}")
                        remp_edit = remplaçants_interactif(f"edit_match_{mid}", [j["Nom"] for p in POSTES_ORDER for j in match["details"].get(p, []) if j])
                        render_replacements(remp_edit, st.session_state.players)
                        if st.button("Mettre à jour la compo", key=f"maj_compo_{mid}"):
                            match["details"] = st.session_state.get(f"terrain_match_{mid}", match["details"])
                            match["remplacants"] = remp_edit
                            save_all()
                            st.success("Composition du match mise à jour.")
                    match_ended = st.checkbox("Match terminé", value=match.get("noted", False), key=f"ended_{mid}")
                    if match_ended and not match.get("noted", False):
                        st.write("### Saisie des stats du match")
                        joueurs_all = [j['Nom'] for p in POSTES_ORDER for j in match["details"].get(p, []) if j]
                        score = st.text_input("Score (ex: 2-1)", key=f"score_{mid}")

                        buteurs_qte = {}
                        st.write("#### Buteurs")
                        for nom in joueurs_all:
                            q = st.number_input(f"{nom} - Buts", min_value=0, max_value=10, value=0, step=1, key=f"but_{mid}_{nom}")
                            if q > 0:
                                buteurs_qte[nom] = q

                        passeurs_qte = {}
                        st.write("#### Passeurs")
                        for nom in joueurs_all:
                            q = st.number_input(f"{nom} - Passes", min_value=0, max_value=10, value=0, step=1, key=f"pass_{mid}_{nom}")
                            if q > 0:
                                passeurs_qte[nom] = q

                        cj_qte = {}
                        st.write("#### Cartons jaunes")
                        for nom in joueurs_all:
                            q = st.number_input(f"{nom} - Cartons jaunes", min_value=0, max_value=5, value=0, step=1, key=f"cj_{mid}_{nom}")
                            if q > 0:
                                cj_qte[nom] = q

                        cr_qte = {}
                        st.write("#### Cartons rouges")
                        for nom in joueurs_all:
                            q = st.number_input(f"{nom} - Cartons rouges", min_value=0, max_value=3, value=0, step=1, key=f"cr_{mid}_{nom}")
                            if q > 0:
                                cr_qte[nom] = q

                        notes = {}
                        st.write("#### Notes des joueurs")
                        for nom in joueurs_all:
                            notes[nom] = st.slider(f"Note pour {nom}", min_value=1, max_value=10, value=6, step=1, key=f"note_{mid}_{nom}")

                        homme_du_match = st.selectbox("Homme du match", [""] + joueurs_all, key=f"hdm_{mid}")

                        if st.button("Valider le match", key=f"valide_{mid}"):
                            match["score"] = score
                            match["events"] = {
                                "buteurs": buteurs_qte,
                                "passeurs": passeurs_qte,
                                "cartons_jaunes": cj_qte,
                                "cartons_rouges": cr_qte,
                                "notes": notes
                            }
                            match["noted"] = True
                            match["homme_du_match"] = homme_du_match
                            save_all()
                            st.success("Stats du match enregistrées !")
                            st.experimental_rerun()
                    if st.button(f"Supprimer ce match", key=f"suppr_match_{mid}"):
                        del st.session_state.matches[mid]
                        save_all()
                        st.experimental_rerun()

# --- PAGE SAUVEGARDE / IMPORT ---
elif menu == "Sauvegarde / Import":
    st.title("Sauvegarde et importation manuelles des données")
    st.info("Téléchargez ou importez toutes vos données (joueurs, compos, matchs) en un seul fichier.")
    download_upload_buttons()
