import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

DATA_FILE = "afcdata.json"
PLAYER_COLS = ["Nom", "Poste", "Infos"]
PLAYER_DEFAULTS = {"Nom": "", "Poste": "G", "Infos": ""}

# --- FORMATIONS ET ICONES ---
FORMATION = {
    "4-4-2": {"G": 1, "D": 4, "M": 4, "A": 2},
    "4-3-3": {"G": 1, "D": 4, "M": 3, "A": 3},
    "3-5-2": {"G": 1, "D": 3, "M": 5, "A": 2},
    "3-4-3": {"G": 1, "D": 3, "M": 4, "A": 3},
    "5-3-2": {"G": 1, "D": 5, "M": 3, "A": 2},
    "4-2-3-1": {"G": 1, "D": 4, "M": 2, "MO": 3, "A": 1},  # Nouvelle formation
}
POSTES_ORDER = ["G", "D", "M", "MO", "A"]  # Ajout MO pour 4-2-3-1
ICONS = {"G": "🧤", "D": "🛡️", "M": "🦶", "MO": "💡", "A": "⚽"}
DEFAULT_FORMATION = "4-4-2"
MAX_REMPLACANTS = 5

# --- UTILITAIRES PERSISTANCE ---
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

# --- CALCUL DYNAMIQUE DES STATS JOUEURS ---
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

# --- INITIALISATION SESSION ---
if "players" not in st.session_state:
    reload_all()
if "lineups" not in st.session_state:
    reload_all()
if "matches" not in st.session_state:
    reload_all()
if "formation" not in st.session_state:
    st.session_state.formation = DEFAULT_FORMATION

# --- FONCTIONS DE DOWNLOAD/UPLOAD ---
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

# --- REMPLAÇANTS INTERACTIFS ---
def remplaçants_interactif_v2(key, titulaires):
    if f"remp_{key}" not in st.session_state:
        st.session_state[f"remp_{key}"] = [None] * MAX_REMPLACANTS
    remps = st.session_state[f"remp_{key}"]

    all_players = [n for n in st.session_state.players["Nom"] if n not in titulaires]
    for i in range(MAX_REMPLACANTS):
        col = st.columns(MAX_REMPLACANTS)[i]
        with col:
            current = remps[i]
            options = [p for p in all_players if p not in remps or p == current]
            value = current if current in options else None
            choix = st.selectbox(
                f"Remplaçant {i+1}",
                [""] + options,
                index=(options.index(current)+1) if current in options else 0,
                key=f"remp_choice_{key}_{i}"
            )
            if choix:
                remps[i] = choix
            else:
                remps[i] = None
    st.session_state[f"remp_{key}"] = remps
    # Affichage visuel sous forme de badge
    render_remplaçants(remps)
    return [r for r in remps if r]

def render_remplaçants(remplaçants_list):
    st.markdown("### Remplaçants")
    html = '<div style="display:flex;justify-content:center;gap:20px;">'
    for r in remplaçants_list:
        if r:
            html += f'<div style="text-align:center;background:#2b3b22;border-radius:12px;padding:8px 16px;color:#fff;min-width:80px;"><b>{r}</b></div>'
        else:
            html += f'<div style="text-align:center;background:#444b44;border-radius:12px;padding:8px 16px;color:#bbb;min-width:80px;">+</div>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

# --- TERRAIN INTERACTIF V2 (NOUVEAU VISUEL) ---
def terrain_interactif_v2(formation, terrain_key, show_export=True):
    # Init du terrain selon formation
    def terrain_init(formation):
        return {poste: [None for _ in range(FORMATION[formation].get(poste, 0))] for poste in POSTES_ORDER}

    # Init session
    if terrain_key not in st.session_state or st.session_state.get(f"formation_{terrain_key}", None) != formation:
        st.session_state[terrain_key] = terrain_init(formation)
        st.session_state[f"formation_{terrain_key}"] = formation
    terrain = st.session_state[terrain_key]

    # Pour le formulaire d’édition
    def joueur_deja_sur_terrain():
        return set(
            j["Nom"]
            for p in POSTES_ORDER
            for j in terrain.get(p, [])
            if j and isinstance(j, dict) and j.get("Nom")
        )

    # Layout pour affichage vertical
    layout = []
    for poste in POSTES_ORDER:
        if poste in FORMATION[formation]:
            layout.append((poste, FORMATION[formation][poste]))

    # Rendu du terrain (HTML/CSS)
    pitch_html = '<div id="pitch" style="display:flex;flex-direction:column;align-items:center;background:#228B22;border-radius:30px;padding:24px 4px;width:350px;min-height:600px;margin:auto;border:4px solid #fff;box-shadow:0 0 12px #444;">'
    for poste, n in layout:
        pitch_html += '<div style="display:flex;justify-content:center;margin:14px 0;">'
        for i in range(n):
            joueur = terrain[poste][i]
            if joueur:
                nom = joueur["Nom"]
                numero = joueur.get("Numero", "")
                is_captain = joueur.get("Capitaine", False)
                cap = ' <span style="color:gold;font-weight:bold;">(C)</span>' if is_captain else ""
                pitch_html += f'''
                <div style="margin:0 16px;text-align:center;min-width:70px;">
                    <div style="font-size:2em">{ICONS[poste]}</div>
                    <div style="font-size:1em;font-weight:bold;color:#fff;">{nom}</div>
                    <div style="color:#ff0;font-size:0.9em;">#{numero}{cap}</div>
                </div>
                '''
            else:
                pitch_html += f'''
                <div style="margin:0 16px;text-align:center;min-width:70px;">
                    <div style="font-size:2em;filter:opacity(0.5);">{ICONS[poste]}</div>
                    <div style="color:#fff;font-size:0.9em;">{poste}{i+1}</div>
                </div>
                '''
        pitch_html += '</div>'
    pitch_html += '</div>'

    st.markdown(pitch_html, unsafe_allow_html=True)

    # Boutons pour ajouter/modifier chaque poste (sous le terrain)
    st.markdown("#### Modifier/Ajouter un joueur sur le terrain")
    for poste, n in layout:
        cols = st.columns(n)
        for i in range(n):
            joueur = terrain[poste][i]
            label = f"{ICONS[poste]} {poste}{i+1} : {'Modifier' if joueur else 'Ajouter'}"
            if cols[i].button(label, key=f"btn_{terrain_key}_{poste}_{i}"):
                st.session_state[f"edit_{terrain_key}"] = (poste, i)

    # Formulaire de sélection/édition joueur
    edit_key = f"edit_{terrain_key}"
    if edit_key in st.session_state:
        poste, idx = st.session_state[edit_key]
        st.markdown(f"---\n### Modifier poste {ICONS[poste]} {poste}{idx+1}")
        joueurs_sur_terrain = joueur_deja_sur_terrain()
        joueur_courant = terrain[poste][idx]["Nom"] if terrain[poste][idx] else None
        if joueur_courant:
            joueurs_sur_terrain = joueurs_sur_terrain - {joueur_courant}
        all_options = st.session_state.players["Nom"].tolist()
        options = [n for n in all_options if n not in joueurs_sur_terrain]
        choix = st.selectbox("Choisir un joueur", [""] + options, key=f"choix_{terrain_key}_{poste}_{idx}")
        numero = st.number_input("Numéro de maillot", min_value=1, max_value=99, value=terrain[poste][idx]["Numero"] if terrain[poste][idx] else 10, key=f"num_{terrain_key}_{poste}_{idx}")
        cap = st.checkbox("Capitaine", value=terrain[poste][idx]["Capitaine"] if terrain[poste][idx] else False, key=f"cap_{terrain_key}_{poste}_{idx}")

        # Un seul capitaine possible : décocher les autres si cap==True
        if cap:
            for p in POSTES_ORDER:
                for j, joueur in enumerate(terrain[p]):
                    if joueur and joueur.get("Capitaine") and (p != poste or j != idx):
                        joueur["Capitaine"] = False
        if st.button("Valider ce joueur", key=f"valider_{terrain_key}_{poste}_{idx}"):
            if choix:
                terrain[poste][idx] = {
                    "Nom": choix,
                    "Numero": numero,
                    "Capitaine": cap
                }
                del st.session_state[edit_key]
                st.session_state[terrain_key] = terrain
                st.experimental_rerun()
        if st.button("Retirer ce joueur", key=f"retirer_{terrain_key}_{poste}_{idx}"):
            terrain[poste][idx] = None
            del st.session_state[edit_key]
            st.session_state[terrain_key] = terrain
            st.experimental_rerun()

    # Export du visuel : capture d’écran utilisateur
    if show_export:
        st.markdown("### Export du visuel")
        st.info("Pour partager la compo, fais une capture d’écran du terrain ci-dessus (sur mobile : bouton de capture ; sur PC : Impr écran ou outil de capture)")

    # Affichage synthétique texte (facultatif)
    st.markdown("**Résumé de la composition :**")
    for poste in POSTES_ORDER:
        joueurs = [
            f"{ICONS[poste]} {j['Nom']} (#{j.get('Numero', '')}){' (C)' if j.get('Capitaine') else ''}"
            for j in terrain.get(poste, []) if j
        ]
        if joueurs:
            st.write(f"**{poste}** : " + ", ".join(joueurs))
    st.session_state[terrain_key] = terrain
    st.session_state[f"formation_{terrain_key}"] = formation
    return terrain

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
        terrain = terrain_interactif_v2(formation, "terrain_create_compo")
        titulaires = [j["Nom"] for p in POSTES_ORDER for j in terrain.get(p, []) if j]
        remplaçants = remplaçants_interactif_v2("create_compo", titulaires)
        if st.button("Sauvegarder la composition"):
            if not nom_compo.strip():
                st.warning("Veuillez donner un nom à la composition.")
            else:
                lineup = {
                    "formation": formation,
                    "details": terrain,
                    "remplacants": remplaçants
                }
                st.session_state.lineups[nom_compo] = lineup
                save_all()
                st.success("Composition sauvegardée !")
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
                    st.write("**Remplaçants :** " + ", ".join(compo.get("remplacants", [])))
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
            terrain = terrain_interactif_v2(formation, "terrain_new_match")

        # Titulaires pour éviter doublons remplaçants
        tous_titulaires = [j["Nom"] for p in POSTES_ORDER for j in st.session_state.get("terrain_new_match", terrain).get(p, []) if j]
        remplaçants = remplaçants_interactif_v2("new_match", tous_titulaires)

        # Enregistrer la compo depuis la création d'un match
        if st.button("Enregistrer cette compo"):
            name_compo = st.text_input("Nom pour la compo à enregistrer", value=nom_match)
            if name_compo:
                lineup = {
                    "formation": formation,
                    "details": st.session_state.get("terrain_new_match", terrain),
                    "remplacants": remplaçants
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

                    if not match.get("noted", False):
                        st.session_state[f"formation_terrain_match_{mid}"] = match["formation"]
                        st.session_state[f"terrain_match_{mid}"] = match["details"]
                        terrain = terrain_interactif_v2(match["formation"], f"terrain_match_{mid}", show_export=False)
                        remp_edit = remplaçants_interactif_v2(f"edit_match_{mid}", [j["Nom"] for p in POSTES_ORDER for j in match["details"].get(p, []) if j])
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
