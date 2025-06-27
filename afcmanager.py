import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime

DATA_FILE = "afcdata.json"
PLAYER_COLS = ["Nom", "Poste", "Infos"]
STATS_COLS = [
    "Buts", "Passes décisives", "Buts + Passes", "Décisif par match",
    "Cartons jaunes", "Cartons rouges", "Sélections", "Titularisations",
    "Note générale", "Homme du match"
]
ALL_COLS = PLAYER_COLS + STATS_COLS

FORMATION = {
    "4-4-2": [("G", 50, 95), ("D1", 10, 80), ("D2", 35, 80), ("D3", 65, 80), ("D4", 90, 80),
              ("M1", 15, 60), ("M2", 35, 60), ("M3", 65, 60), ("M4", 85, 60),
              ("A1", 35, 35), ("A2", 65, 35)],
    "4-3-3": [("G", 50, 95), ("D1", 10, 80), ("D2", 35, 80), ("D3", 65, 80), ("D4", 90, 80),
              ("M1", 28, 60), ("M2", 50, 60), ("M3", 72, 60),
              ("A1", 20, 35), ("A2", 50, 30), ("A3", 80, 35)],
    "3-5-2": [("G", 50, 95), ("D1", 20, 80), ("D2", 50, 80), ("D3", 80, 80),
              ("M1", 10, 60), ("M2", 30, 60), ("M3", 50, 55), ("M4", 70, 60), ("M5", 90, 60),
              ("A1", 38, 35), ("A2", 62, 35)],
    "3-4-3": [("G", 50, 95), ("D1", 20, 80), ("D2", 50, 80), ("D3", 80, 80),
              ("M1", 20, 60), ("M2", 40, 60), ("M3", 60, 60), ("M4", 80, 60),
              ("A1", 25, 35), ("A2", 50, 30), ("A3", 75, 35)],
    "5-3-2": [("G", 50, 95), ("D1", 5, 80), ("D2", 25, 80), ("D3", 50, 80), ("D4", 75, 80), ("D5", 95, 80),
              ("M1", 30, 60), ("M2", 50, 60), ("M3", 70, 60),
              ("A1", 38, 35), ("A2", 62, 35)],
    "4-2-3-1": [("G", 50, 95), ("D1", 10, 80), ("D2", 35, 80), ("D3", 65, 80), ("D4", 90, 80),
                ("M1", 32, 65), ("M2", 68, 65),
                ("MO1", 22, 50), ("MO2", 50, 45), ("MO3", 78, 50),
                ("A1", 50, 30)],
}
POSTES_ORDER = ["G", "D", "M", "MO", "A"]
DEFAULT_FORMATION = "4-4-2"

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

def compute_player_stats(joueur_nom):
    buts = passes = cj = cr = selections = titularisations = note_sum = note_count = hdm = 0
    for match in st.session_state.matches.values():
        details = match.get("details", {})
        joueurs = [j for p in POSTES_ORDER for j in details.get(p, []) if j and isinstance(j, dict) and j.get("Nom") == joueur_nom]
        is_titulaire = bool(joueurs)
        if is_titulaire or joueur_nom in [r.get("Nom") for r in match.get("remplacants", []) if isinstance(r, dict) and r.get("Nom")]:
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

# ---------- INITIALISATION SESSION ----------
if "players" not in st.session_state:
    reload_all()
if "lineups" not in st.session_state:
    reload_all()
if "matches" not in st.session_state:
    reload_all()
if "formation" not in st.session_state:
    st.session_state.formation = DEFAULT_FORMATION

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
            st.success("Données importées ! Cliquez sur Rafraîchir la base de données si elle n'est pas à jour.")
        except Exception as e:
            st.error(f"Erreur à l'import : {e}")

def terrain_viz_simple(formation, titulaires, remplaçants, captain_name):
    titulaires = titulaires or []
    remplaçants = remplaçants or []
    postes = FORMATION[formation]
    st.markdown('''
    <div style="position:relative;width:100%;max-width:480px;aspect-ratio:2/3;margin:auto;background:linear-gradient(180deg,#4db367 0%,#245c32 100%);border-radius:30px;border:3px solid #fff;overflow:hidden;">
    ''', unsafe_allow_html=True)
    html = ""
    idx = 0
    for poste, x, y in postes:
        joueur = titulaires[idx] if idx < len(titulaires) else None
        if joueur and isinstance(joueur, dict) and joueur.get("Nom"):
            is_cap = joueur.get("Nom") == captain_name
            html += f'''
            <div style="position:absolute;left:{x}%;top:{y}%;width:13%;min-width:50px;text-align:center;transform:translate(-50%,-50%);">
                <div style="font-size:0.85em;color:#FFD700;text-shadow:0 1px 2px #000a;">{poste}</div>
                <div style="background:#1976D2;color:#fff;width:3.5em;height:3.5em;border-radius:10px;border:3px solid {'#FFD700' if is_cap else '#fff'};display:flex;align-items:center;justify-content:center;font-weight:bold;font-size:1.4em;margin:auto;position:relative;">
                    {joueur.get("Numero", "")}
                    {('<span style="position:absolute;top:-13px;right:-12px;background:#FFD700;color:#000;padding:2px 6px;border-radius:10px;font-size:0.8em;font-weight:bold;">C</span>' if is_cap else '')}
                </div>
                <div style="font-size:0.95em;color:#fff;text-shadow:0 1px 2px #000a;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">{joueur.get("Nom")}</div>
            </div>
            '''
        idx += 1
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)
    remp_aff = [f'{r.get("Nom")} (#{r.get("Numero")})'
                for r in remplaçants
                if isinstance(r, dict) and r.get("Nom")]
    if remp_aff:
        st.markdown("**Remplaçants** : " + ", ".join(remp_aff))

def choix_joueurs_interface(formation, key_prefix):
    postes = []
    for poste, n in FORMATION[formation].items() if isinstance(FORMATION[formation], dict) else []:
        for i in range(n):
            postes.append(f"{poste}{i+1}")
    if not postes:
        postes = [p[0] for p in FORMATION[formation]]
    all_joueurs = st.session_state.players["Nom"].tolist()
    titulaires = []
    selected = set()
    for idx, label in enumerate(postes):
        choix = st.selectbox(label, [""] + [j for j in all_joueurs if j not in selected],
                             key=f"{key_prefix}_poste_{label}")
        if choix: selected.add(choix)
        numero = st.number_input(f"Numéro pour {choix}", 1, 99, 10+idx, key=f"{key_prefix}_num_{label}")
        titulaires.append({"Nom": choix, "Numero": numero} if choix else None)
    noms_titulaires = [t["Nom"] for t in titulaires if t and t["Nom"]]
    capitaine = st.selectbox("Capitaine", noms_titulaires, key=f"{key_prefix}_capitaine") if noms_titulaires else ""
    remplaçants = []
    selected_remp = set(noms_titulaires)
    for i in range(5):
        choix = st.selectbox(f"Remplaçant {i+1}", [""] + [j for j in all_joueurs if j not in selected_remp],
                             key=f"{key_prefix}_remp_{i}")
        if choix: selected_remp.add(choix)
        numero = st.number_input(f"Numéro pour {choix}", 1, 99, 16+i, key=f"{key_prefix}_num_remp_{i}")
        remplaçants.append({"Nom": choix, "Numero": numero} if choix else None)
    return titulaires, remplaçants, capitaine

tab_labels = ["Database", "Compositions", "Matchs", "Sauvegarde"]
tab_database, tab_compositions, tab_matchs, tab_sauvegarde = st.tabs(tab_labels)

with tab_database:
    st.title("Base de données joueurs (édition + stats)")
    if st.button("Rafraîchir la base de données"):
        reload_all()
        st.success("Base rechargée depuis le fichier !")
    base_df = st.session_state.players.copy()
    all_rows = []
    for _, row in base_df.iterrows():
        s = compute_player_stats(row["Nom"])
        all_rows.append({**row, **s})
    merged_df = pd.DataFrame(all_rows)
    for col in PLAYER_COLS + STATS_COLS:
        if col not in merged_df.columns:
            merged_df[col] = ""
    if merged_df.empty:
        merged_df = pd.DataFrame(columns=ALL_COLS)
    # Edition seulement des colonnes joueurs, stats désactivées
    editable_cols = PLAYER_COLS
    disabled_cols = [col for col in merged_df.columns if col not in editable_cols]
    edited_df = st.data_editor(
        merged_df,
        column_config={col: st.column_config.Column(disabled=True) for col in disabled_cols},
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

with tab_compositions:
    st.title("Gestion des compositions")
    tab1, tab2 = st.tabs(["Créer une composition", "Mes compositions"])
    with tab1:
        nom_compo = st.text_input("Nom de la composition")
        formation = st.selectbox("Formation", list(FORMATION.keys()), key="formation_create")
        titulaires, remplaçants, capitaine = choix_joueurs_interface(formation, "create")
        terrain_viz_simple(formation, titulaires, remplaçants, capitaine)
        if st.button("Sauvegarder la composition"):
            if not nom_compo.strip():
                st.warning("Veuillez donner un nom à la composition.")
            else:
                lineup = {
                    "formation": formation,
                    "titulaires": titulaires,
                    "remplacants": remplaçants,
                    "capitaine": capitaine
                }
                st.session_state.lineups[nom_compo] = lineup
                save_all()
                st.success("Composition sauvegardée !")
    with tab2:
        if not st.session_state.lineups:
            st.info("Aucune composition enregistrée.")
        else:
            for nom, compo in st.session_state.lineups.items():
                with st.expander(f"{nom} – {compo.get('formation', '')}"):
                    terrain_viz_simple(
                        compo.get("formation", DEFAULT_FORMATION),
                        compo.get("titulaires", []) or [],
                        compo.get("remplacants", []) or [],
                        compo.get("capitaine", "")
                    )
                    col1, col2 = st.columns(2)
                    if col1.button(f"Supprimer {nom}", key=f"suppr_{nom}"):
                        del st.session_state.lineups[nom]
                        save_all()
                        st.experimental_rerun()

with tab_matchs:
    st.title("Gestion des matchs")
    tab1, tab2 = st.tabs(["Créer un match", "Mes matchs"])
    with tab1:
        type_match = st.selectbox("Type de match", ["Championnat", "Coupe"])
        adversaire = st.text_input("Nom de l'adversaire")
        date = st.date_input("Date du match", value=datetime.today())
        heure = st.time_input("Heure du match")
        lieu = st.text_input("Lieu")
        nom_match = st.text_input("Nom du match", value=f"{date.strftime('%Y-%m-%d')} vs {adversaire}" if adversaire else f"{date.strftime('%Y-%m-%d')}")
        formation = st.selectbox("Formation", list(FORMATION.keys()), key="formation_match")
        titulaires, remplaçants, capitaine = choix_joueurs_interface(formation, "match")
        terrain_viz_simple(formation, titulaires, remplaçants, capitaine)
        if st.button("Enregistrer le match"):
            st.session_state.matches[nom_match] = {
                "type": type_match,
                "adversaire": adversaire,
                "date": str(date),
                "heure": str(heure),
                "lieu": lieu,
                "formation": formation,
                "titulaires": titulaires,
                "remplacants": remplaçants,
                "capitaine": capitaine,
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
                with st.expander(f"{match.get('date', '')} {match.get('heure', '')} vs {match.get('adversaire', '')} ({match.get('type', '')})"):
                    terrain_viz_simple(
                        match.get("formation", DEFAULT_FORMATION),
                        match.get("titulaires", []) or [],
                        match.get("remplacants", []) or [],
                        match.get("capitaine", "")
                    )
                    st.write(f"**Lieu :** {match.get('lieu', '')}")
                    if st.button(f"Supprimer ce match", key=f"suppr_match_{mid}"):
                        del st.session_state.matches[mid]
                        save_all()
                        st.experimental_rerun()

with tab_sauvegarde:
    st.title("Sauvegarde et importation manuelles des données")
    st.info("Téléchargez ou importez toutes vos données (joueurs, compos, matchs) en un seul fichier.")
    download_upload_buttons()