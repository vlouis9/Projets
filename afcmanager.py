import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

# --- CONSTANTES ---
DATA_FILE = "afcdata.json"
PLAYER_COLS = ["Nom", "Poste", "Infos"]
PLAYER_DEFAULTS = {"Nom": "", "Poste": "G", "Infos": ""}

FORMATION = {
    "4-2-3-1": {"G": 1, "D": 4, "M": 5, "A": 1},  # Nouvelle formation par défaut
    "4-4-2": {"G": 1, "D": 4, "M": 4, "A": 2},
    "4-3-3": {"G": 1, "D": 4, "M": 3, "A": 3},
    "3-5-2": {"G": 1, "D": 3, "M": 5, "A": 2},
    "3-4-3": {"G": 1, "D": 3, "M": 4, "A": 3},
    "5-3-2": {"G": 1, "D": 5, "M": 3, "A": 2},
}
POSTES_ORDER = ["G", "D", "M", "A"]
DEFAULT_FORMATION = "4-2-3-1"  # Formation par défaut
MAX_REMPLACANTS = 5

# --- Utilitaires persistance ---
def save_all():
    """Sauvegarde toutes les données dans le fichier JSON"""
    data = {
        "players": st.session_state.players.to_dict(orient="records"),
        "lineups": st.session_state.lineups,
        "matches": st.session_state.matches,
    }
    with open(DATA_FILE, "w") as f:
        json.dump(data, f, indent=2)

def reload_all():
    """Charge toutes les données depuis le fichier JSON"""
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

def compute_player_stats(joueur_nom):
    """Calcule les statistiques d'un joueur"""
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

def terrain_init(formation):
    """Initialise un terrain vide avec une formation donnée"""
    return {poste: [None for _ in range(FORMATION[formation][poste])] for poste in POSTES_ORDER}

def terrain_interactif(formation, terrain_key):
    """Interface pour la gestion du terrain et des joueurs"""
    if terrain_key not in st.session_state or st.session_state.get(f"formation_{terrain_key}", None) != formation:
        st.session_state[terrain_key] = terrain_init(formation)
        st.session_state[f"formation_{terrain_key}"] = formation
    terrain = st.session_state[terrain_key]

    def joueur_deja_sur_terrain():
        return set(
            j["Nom"]
            for p in POSTES_ORDER
            for j in terrain.get(p, [])
            if j and isinstance(j, dict) and j.get("Nom")
        )

    def poste_buttons(poste, n):
        cols = st.columns(n)
        for i in range(n):
            joueur = terrain[poste][i]
            if joueur:
                label = f"{joueur['Nom']} (#{joueur.get('Numero', '')}){' (C)' if joueur.get('Capitaine') else ''}"
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
        joueurs_sur_terrain = joueur_deja_sur_terrain()
        joueur_courant = terrain[poste][idx]["Nom"] if terrain[poste][idx] else None
        if joueur_courant:
            joueurs_sur_terrain = joueurs_sur_terrain - {joueur_courant}
        all_options = st.session_state.players["Nom"].tolist()
        options = [n for n in all_options if n not in joueurs_sur_terrain]
        choix = st.selectbox("Choisir un joueur", [""] + options, key=f"choix_{terrain_key}_{poste}_{idx}")
        numero = st.number_input("Numéro de maillot", min_value=1, max_value=99, value=terrain[poste][idx]["Numero"] if terrain[poste][idx] else 10, key=f"num_{terrain_key}_{poste}_{idx}")
        capitaine = st.checkbox("Capitaine", value=terrain[poste][idx]["Capitaine"] if terrain[poste][idx] else False, key=f"cap_{terrain_key}_{poste}_{idx}")
        if st.button("Valider ce joueur", key=f"valider_{terrain_key}_{poste}_{idx}"):
            if choix:
                terrain[poste][idx] = {
                    "Nom": choix,
                    "Numero": numero,
                    "Capitaine": capitaine
                }
                del st.session_state[edit_key]
                st.session_state[terrain_key] = terrain
                st.experimental_rerun()
        if st.button("Retirer ce joueur", key=f"retirer_{terrain_key}_{poste}_{idx}"):
            terrain[poste][idx] = None
            del st.session_state[edit_key]
            st.session_state[terrain_key] = terrain
            st.experimental_rerun()

    st.markdown("**Composition actuelle :**")
    for poste in POSTES_ORDER:
        joueurs = [
            f"{j['Nom']} (#{j.get('Numero', '')}){' (C)' if j.get('Capitaine') else ''}"
            for j in terrain.get(poste, []) if j
        ]
        st.write(f"**{poste}** : {', '.join(joueurs) if joueurs else 'Aucun'}")
    st.session_state[terrain_key] = terrain
    st.session_state[f"formation_{terrain_key}"] = formation
    return terrain

def remplaçants_interactif(key, titulaires):
    """Interface pour la sélection des remplaçants"""
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

# --- Initialisation session ---
if "players" not in st.session_state:
    reload_all()
if "lineups" not in st.session_state:
    reload_all()
if "matches" not in st.session_state:
    reload_all()
if "formation" not in st.session_state:
    st.session_state.formation = DEFAULT_FORMATION

def download_upload_buttons():
    """Interface pour l'import/export des données"""
    st.download_button(
        label="📥 Télécharger données (JSON)",
        data=json.dumps({
            "players": st.session_state.players.to_dict(orient="records"),
            "lineups": st.session_state.lineups,
            "matches": st.session_state.matches,
        }, indent=2),
        file_name=DATA_FILE,
        mime="application/json"
    )
    up_json = st.file_uploader("📤 Importer données (JSON)", type="json", key="upload_all")
    if up_json:
        try:
            data = json.load(up_json)
            st.session_state.players = pd.DataFrame(data.get("players", []))
            st.session_state.lineups = data.get("lineups", {})
            st.session_state.matches = data.get("matches", {})
            st.success("✅ Données importées avec succès!")
        except Exception as e:
            st.error(f"❌ Erreur à l'import : {e}")

# --- INTERFACE PRINCIPALE ---
st.sidebar.title("⚽ Gestion Équipe AFC")

# Import/Export exclusivement dans la barre latérale
with st.sidebar:
    st.markdown("---")
    with st.expander("📥 Import/Export des données"):
        download_upload_buttons()
    st.markdown("---")

# Corps principal avec onglets
tab1, tab2, tab3 = st.tabs(["Database", "Compositions", "Matchs"])

# --- DATABASE ---
with tab1:
    st.title("Base de données joueurs")
    st.markdown("Vous pouvez **éditer, supprimer ou ajouter** des joueurs directement dans le tableau ci-dessous.")
    
    # Création du dataframe combiné avec stats
    stats_data = []
    for _, row in st.session_state.players.iterrows():
        s = compute_player_stats(row["Nom"])
        stats_data.append({**row, **s})
    
    combined_df = pd.DataFrame(stats_data, columns=[
        "Nom", "Poste", "Infos", "Buts", "Passes décisives", 
        "Buts + Passes", "Décisif par match", "Cartons jaunes", 
        "Cartons rouges", "Sélections", "Titularisations", 
        "Note générale", "Homme du match"
    ])
    
    # Affichage et édition du tableau unique
    edited_df = st.data_editor(
        combined_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Nom": st.column_config.TextColumn(required=True),
            "Poste": st.column_config.SelectboxColumn(
                options=POSTES_ORDER,
                required=True,
                default="G"
            ),
            "Infos": st.column_config.TextColumn(),
            "Buts": st.column_config.NumberColumn(disabled=True),
            "Passes décisives": st.column_config.NumberColumn(disabled=True),
            "Buts + Passes": st.column_config.NumberColumn(disabled=True),
            "Décisif par match": st.column_config.NumberColumn(disabled=True),
            "Cartons jaunes": st.column_config.NumberColumn(disabled=True),
            "Cartons rouges": st.column_config.NumberColumn(disabled=True),
            "Sélections": st.column_config.NumberColumn(disabled=True),
            "Titularisations": st.column_config.NumberColumn(disabled=True),
            "Note générale": st.column_config.NumberColumn(disabled=True),
            "Homme du match": st.column_config.NumberColumn(disabled=True)
        },
        key="data_edit"
    )
    
    if st.button("Sauvegarder les modifications"):
        edited_df = edited_df.fillna("")
        edited_df = edited_df[edited_df["Nom"].str.strip() != ""]
        # Ne garder que les colonnes de base pour la sauvegarde
        st.session_state.players = edited_df[PLAYER_COLS]
        save_all()
        reload_all()
        st.success("Base de joueurs mise à jour !")
    st.caption("Pour supprimer une ligne, videz le nom du joueur puis cliquez sur Sauvegarder.")

# --- COMPOSITIONS ---
with tab2:
    st.title("Gestion des compositions")
    subtab1, subtab2 = st.tabs(["Créer une composition", "Mes compositions"])
    
    # Edition/Création
    with subtab1:
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

    # Liste/Edition
    with subtab2:
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
with tab3:
    st.title("Gestion des matchs")
    subtab1, subtab2 = st.tabs(["Créer un match", "Mes matchs"])

    # Création de match
    with subtab1:
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

    # Liste des matchs
    with subtab2:
        if not st.session_state.matches:
            st.info("Aucun match enregistré.")
        else:
            for mid, match in st.session_state.matches.items():
                with st.expander(f"{match['date']} {match['heure']} vs {match['adversaire']} ({match['type']})"):
                    statut = "Terminé" if match.get("noted", False) else "En cours"
                    st.write(f"**Statut :** {statut}")
                    if match.get("noted", False):
                        # En-tête avec score
                        score_col1, score_col2, score_col3 = st.columns([2,1,2])
                        with score_col1:
                            st.markdown(f"### {match['adversaire']}")
                        with score_col2:
                            st.markdown(f"### {match.get('score','')}")
                        with score_col3:
                            st.markdown("### AFC")
                        
                        st.markdown("---")
                        
                        # Statistiques du match en colonnes
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("#### 📊 Stats du match")
                            ev = match.get("events", {})
                            
                            # Buteurs et passeurs côte à côte
                            but_col, pass_col = st.columns(2)
                            with but_col:
                                st.markdown("**⚽ Buteurs**")
                                for nom, nb in ev.get("buteurs", {}).items():
                                    st.markdown(f"- {nom} ({nb})")
                            with pass_col:
                                st.markdown("**👟 Passeurs**")
                                for nom, nb in ev.get("passeurs", {}).items():
                                    st.markdown(f"- {nom} ({nb})")
                        
                        with col2:
                            st.markdown("#### 🎯 Performance")
                            # Homme du match avec mise en valeur
                            st.markdown(f"**🏆 Homme du match :** {match.get('homme_du_match','')}")
                            
                            # Notes moyennes triées
                            notes = ev.get("notes", {})
                            if notes:
                                st.markdown("**⭐ Meilleures notes:**")
                                sorted_notes = sorted(notes.items(), key=lambda x: x[1], reverse=True)
                                for nom, note in sorted_notes[:3]:  # Top 3 des notes
                                    st.markdown(f"- {nom}: {note}/10")
                        
                        # Section discipline en bas
                        st.markdown("#### 📋 Discipline")
                        disc_col1, disc_col2 = st.columns(2)
                        with disc_col1:
                            st.markdown("**🟨 Cartons jaunes**")
                            for nom, nb in ev.get("cartons_jaunes", {}).items():
                                st.markdown(f"- {nom} ({nb})")
                        with disc_col2:
                            st.markdown("**🟥 Cartons rouges**")
                            for nom, nb in ev.get("cartons_rouges", {}).items():
                                st.markdown(f"- {nom} ({nb})")
                        
                        st.markdown("---")
                        
                        # Composition du match
                        st.markdown("#### 📋 Composition")
                        compo_col1, compo_col2 = st.columns(2)
                        with compo_col1:
                            st.write(f"**Formation:** {match['formation']}")
                            for poste in POSTES_ORDER:
                                joueurs = [
                                    f"{j['Nom']}{' (C)' if j.get('Capitaine') else ''}"
                                    for j in match["details"].get(poste, []) if j
                                ]
                                st.write(f"**{poste}** : {', '.join(joueurs) if joueurs else 'Aucun'}")
                        with compo_col2:
                            st.write("**Remplaçants:**")
                            for remp in match.get("remplacants", []):
                                st.markdown(f"- {remp}")

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
                        terrain = terrain_interactif(match["formation"], f"terrain_match_{mid}")
                        remp_edit = remplaçants_interactif(f"edit_match_{mid}", [j["Nom"] for p in POSTES_ORDER for j in match["details"].get(p, []) if j])
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
                            match["events"] =