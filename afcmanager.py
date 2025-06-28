import streamlit as st
import pandas as pd
import json
import os
import copy
import traceback
from datetime import datetime
import plotly.graph_objects as go

DATA_FILE = "afcdata.json"
PLAYER_COLS = ["Nom", "Poste", "Infos"]

POSTES_DETAILES = {
    "4-2-3-1": [
        ("Gardien", "G"),
        ("Défenseur gauche", "DL"),
        ("Défenseur central gauche", "DCG"),
        ("Défenseur central droit", "DCD"),
        ("Défenseur droit", "DR"),
        ("Milieu récupérateur gauche", "MRG"),
        ("Milieu récupérateur droit", "MRD"),
        ("Milieu offensif gauche", "MOG"),
        ("Milieu offensif axial", "MOA"),
        ("Milieu offensif droit", "MOD"),
        ("Attaquant", "AT"),
    ],
    "4-4-2": [
        ("Gardien", "G"),
        ("Défenseur gauche", "DL"),
        ("Défenseur central gauche", "DCG"),
        ("Défenseur central droit", "DCD"),
        ("Défenseur droit", "DR"),
        ("Milieu gauche", "MLG"),
        ("Milieu central gauche", "MCG"),
        ("Milieu central droit", "MCD"),
        ("Milieu droit", "MLD"),
        ("Attaquant gauche", "ATG"),
        ("Attaquant droit", "ATD"),
    ],
    "4-3-3": [
        ("Gardien", "G"),
        ("Défenseur gauche", "DL"),
        ("Défenseur central gauche", "DCG"),
        ("Défenseur central droit", "DCD"),
        ("Défenseur droit", "DR"),
        ("Milieu gauche", "MLG"),
        ("Milieu axial", "MA"),
        ("Milieu droit", "MLD"),
        ("Ailier gauche", "AIG"),
        ("Avant-centre", "AC"),
        ("Ailier droit", "AID"),
    ],
    "3-5-2": [
        ("Gardien", "G"),
        ("Défenseur gauche", "DL"),
        ("Défenseur central", "DC"),
        ("Défenseur droit", "DR"),
        ("Milieu gauche", "MLG"),
        ("Milieu central gauche", "MCG"),
        ("Milieu axial", "MA"),
        ("Milieu central droit", "MCD"),
        ("Milieu droit", "MLD"),
        ("Attaquant gauche", "ATG"),
        ("Attaquant droit", "ATD"),
    ],
    "3-4-3": [
        ("Gardien", "G"),
        ("Défenseur gauche", "DL"),
        ("Défenseur central", "DC"),
        ("Défenseur droit", "DR"),
        ("Milieu gauche", "MLG"),
        ("Milieu central gauche", "MCG"),
        ("Milieu central droit", "MCD"),
        ("Milieu droit", "MLD"),
        ("Ailier gauche", "AIG"),
        ("Avant-centre", "AC"),
        ("Ailier droit", "AID"),
    ],
    "5-3-2": [
        ("Gardien", "G"),
        ("Latéral gauche", "LG"),
        ("Défenseur central gauche", "DCG"),
        ("Défenseur central", "DC"),
        ("Défenseur central droit", "DCD"),
        ("Latéral droit", "LD"),
        ("Milieu gauche", "MLG"),
        ("Milieu axial", "MA"),
        ("Milieu droit", "MLD"),
        ("Attaquant gauche", "ATG"),
        ("Attaquant droit", "ATD"),
    ],
}
DEFAULT_FORMATION = "4-2-3-1"
MAX_REMPLACANTS = 5

def draw_football_pitch_vertical():
    fig = go.Figure()
    fig.add_shape(type="rect", x0=0, y0=0, x1=68, y1=105, line=dict(width=2, color="#145A32"))
    fig.add_shape(type="rect", x0=13.84, y0=0, x1=68-13.84, y1=16.5, line=dict(width=1, color="#145A32"))
    fig.add_shape(type="rect", x0=13.84, y0=105-16.5, x1=68-13.84, y1=105, line=dict(width=1, color="#145A32"))
    fig.add_shape(type="circle", x0=34-9.15, y0=52.5-9.15, x1=34+9.15, y1=52.5+9.15, line=dict(width=1, color="#145A32"))
    fig.add_shape(type="circle", x0=34-0.4, y0=52.5-0.4, x1=34+0.4, y1=52.5+0.4, fillcolor="#145A32", line=dict(color="#145A32"))
    fig.update_xaxes(showticklabels=False, range=[-5, 73], visible=False)
    fig.update_yaxes(showticklabels=False, range=[-8, 125], visible=False)
    fig.update_layout(
        width=460, height=800, plot_bgcolor="#154734", margin=dict(l=10,r=10,t=10,b=10), showlegend=False
    )
    return fig

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

def terrain_interactif_detaillé(formation, terrain_key):
    postes = POSTES_DETAILES[formation]
    if terrain_key not in st.session_state:
        st.session_state[terrain_key] = {abbr: None for _, abbr in postes}
    terrain = st.session_state[terrain_key]
    for _, abbr in postes:
        if abbr not in terrain:
            terrain[abbr] = None
    col1, col2 = st.columns(2)
    all_selected = [v["Nom"] for v in terrain.values() if isinstance(v, dict) and "Nom" in v and v]
    for idx, (poste_label, abbr) in enumerate(postes):
        col = col1 if idx % 2 == 0 else col2
        current_joueur = terrain[abbr] if isinstance(terrain[abbr], dict) and "Nom" in terrain[abbr] else None
        current_nom = current_joueur["Nom"] if current_joueur else ""
        joueur_options = [""] + [
            n for n in st.session_state.players["Nom"]
            if n and (n == current_nom or n not in all_selected)
        ]
        choix = col.selectbox(
            poste_label,
            joueur_options,
            index=joueur_options.index(current_nom) if current_nom in joueur_options else 0,
            key=f"{terrain_key}_{abbr}"
        )
        if choix:
            joueur_info = st.session_state.players[st.session_state.players["Nom"] == choix].iloc[0].to_dict()
            num = col.text_input(f"Numéro de {choix}", value=current_joueur.get("Numero","") if current_joueur else "", key=f"num_{terrain_key}_{abbr}")
            joueur_info["Numero"] = num
            terrain[abbr] = joueur_info
        else:
            terrain[abbr] = None
    st.session_state[terrain_key] = terrain
    return terrain

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

def plot_lineup_on_pitch_vertical(fig, details, formation, remplaçants=None, capitaine=None):
    postes = POSTES_DETAILES[formation]
    n = len(postes)
    for idx, (poste_label, abbr) in enumerate(postes):
        joueur = details.get(abbr)
        if joueur and isinstance(joueur, dict) and "Nom" in joueur:
            x = 10 + 48 * (idx % 2)
            y = 10 + (idx * (75 // n))
            color = "#ffd700" if capitaine and (joueur["Nom"] == capitaine) else "#0d47a1"
            fig.add_trace(go.Scatter(
                x=[x], y=[y],
                mode="markers+text",
                marker=dict(size=38, color=color, line=dict(width=2, color="white")),
                text=f"{joueur.get('Numero', '')}".strip() if "Numero" in joueur else "",
                textposition="middle center",
                textfont=dict(color="white", size=17, family="Arial Black"),
                hovertext=f"{joueur['Nom']}"+(" (C)" if capitaine and joueur["Nom"] == capitaine else ""),
                hoverinfo="text"
            ))
            fig.add_trace(go.Scatter(
                x=[x], y=[y-4],
                mode="text",
                text=[joueur['Nom'] + (" (C)" if capitaine and joueur["Nom"] == capitaine else "")],
                textfont=dict(color="white", size=13, family="Arial Black"),
                showlegend=False
            ))
    remplaçants = remplaçants or []
    nremp = len(remplaçants)
    if nremp:
        x_start = 34 - 16*nremp/2 + 8
        for idx, remp in enumerate(remplaçants):
            x_r = x_start + idx*16
            fig.add_trace(go.Scatter(
                x=[x_r], y=[-6],
                mode="markers+text",
                marker=dict(size=28, color="#0d47a1", line=dict(width=2, color="white")),
                text="",
                hovertext=remp,
                hoverinfo="text"
            ))
            fig.add_trace(go.Scatter(
                x=[x_r], y=[-11],
                mode="text",
                text=[remp],
                textfont=dict(color="white", size=12, family="Arial Black"),
                showlegend=False
            ))
    return fig

def compute_player_stats(joueur_nom):
    buts = passes = cj = cr = selections = titularisations = note_sum = note_count = hdm = 0
    for match in st.session_state.matches.values():
        details = match.get("details", {})
        joueurs = [j for abbr, j in details.items() if j and isinstance(j, dict) and j.get("Nom") == joueur_nom]
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

if "players" not in st.session_state:
    reload_all()
if "lineups" not in st.session_state:
    reload_all()
if "matches" not in st.session_state:
    reload_all()
if "formation" not in st.session_state:
    st.session_state.formation = DEFAULT_FORMATION

def download_upload_buttons():
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

st.sidebar.title("⚽ Gestion Équipe AFC")
with st.sidebar:
    st.markdown("---")
    with st.expander("📥 Import/Export des données"):
        download_upload_buttons()
    st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Base joueurs", "Compositions", "Matchs"])

# --- BASE DE JOUEURS ---
with tab1:
    st.title("Base de données joueurs")
    st.markdown("Vous pouvez **éditer, supprimer ou ajouter** des joueurs directement dans le tableau ci-dessous.")
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
    edited_df = st.data_editor(
        combined_df,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Nom": st.column_config.TextColumn(required=True),
            "Poste": st.column_config.TextColumn(),
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
        st.session_state.players = edited_df[PLAYER_COLS]
        save_all()
        reload_all()
        st.success("Base de joueurs mise à jour !")
    st.caption("Pour supprimer une ligne, videz le nom du joueur puis cliquez sur Sauvegarder.")

# --- COMPOSITIONS ---
with tab2:
    st.title("Gestion des compositions")
    subtab1, subtab2 = st.tabs(["Créer une composition", "Mes compositions"])
    with subtab1:
        edit_key = "edit_compo"
        edit_compo = st.session_state.get(edit_key, None)
        if edit_compo:
            nom_compo, loaded = edit_compo
            st.info(f"Édition de la compo : {nom_compo}")
            st.session_state["formation_create_compo"] = loaded["formation"]
            st.session_state["terrain_create_compo"] = loaded["details"]
            st.session_state["capitaine_create_compo"] = loaded.get("capitaine", "")
            del st.session_state[edit_key]
        nom_compo = st.text_input("Nom de la composition", value=nom_compo if edit_compo else "")
        formation = st.selectbox(
            "Formation", list(POSTES_DETAILES.keys()),
            index=list(POSTES_DETAILES.keys()).index(st.session_state.get("formation_create_compo", DEFAULT_FORMATION))
        )
        st.session_state["formation_create_compo"] = formation
        terrain = terrain_interactif_detaillé(formation, "terrain_create_compo")
        tous_titulaires = [v["Nom"] for v in terrain.values() if isinstance(v, dict) and "Nom" in v and v]
        remplaçants = remplaçants_interactif("create_compo", tous_titulaires)
        capitaine = st.selectbox("Sélectionner le capitaine", [""] + tous_titulaires, key="capitaine_create_compo")
        fig = draw_football_pitch_vertical()
        fig = plot_lineup_on_pitch_vertical(fig, terrain, formation, remplaçants, capitaine)
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key="fig_create_compo")
        if st.button("Sauvegarder la composition"):
            try:
                lineup = {
                    "formation": formation,
                    "details": copy.deepcopy(terrain),
                    "remplacants": copy.deepcopy(remplaçants),
                    "capitaine": capitaine
                }
                if nom_compo.strip():
                    st.session_state.lineups[nom_compo] = lineup
                    save_all()
                    st.success("Composition sauvegardée !")
                    st.session_state["terrain_create_compo"] = {abbr: None for _, abbr in POSTES_DETAILES[formation]}
                    st.session_state["remp_create_compo"] = [None] * MAX_REMPLACANTS
                    st.session_state["capitaine_create_compo"] = ""
                    st.rerun()
                else:
                    st.warning("Veuillez donner un nom à la composition.")
            except Exception as e:
                st.error(f"Erreur lors de la sauvegarde : {e}")
                st.text(traceback.format_exc())
    with subtab2:
        if not st.session_state.lineups:
            st.info("Aucune composition enregistrée.")
        else:
            for nom, compo in st.session_state.lineups.items():
                with st.expander(f"{nom} – {compo['formation']}"):
                    fig = draw_football_pitch_vertical()
                    fig = plot_lineup_on_pitch_vertical(fig, compo["details"], compo["formation"], compo.get("remplacants", []), compo.get("capitaine"))
                    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"fig_compo_{nom}")
                    col1, col2 = st.columns(2)
                    if col1.button(f"Éditer {nom}", key=f"edit_{nom}"):
                        st.session_state["edit_compo"] = (nom, compo)
                        st.rerun()
                    if col2.button(f"Supprimer {nom}", key=f"suppr_{nom}"):
                        del st.session_state.lineups[nom]
                        save_all()
                        st.rerun()

# --- MATCHS ---
with tab3:
    st.title("Gestion des matchs")
    subtab1, subtab2 = st.tabs(["Créer un match", "Mes matchs"])
    with subtab1:
        if st.button("Réinitialiser la création du match"):
            for k in [
                "terrain_new_match", "formation_new_match",
                "remp_new_match", "nom_match_sugg", "adversaire", "lieu", "capitaine_new_match"
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()
        type_match = st.selectbox("Type de match", ["Championnat", "Coupe"])
        adversaire = st.text_input("Nom de l'adversaire", key="adversaire")
        date = st.date_input("Date du match", value=datetime.today())
        heure = st.time_input("Heure du match")
        lieu = st.text_input("Lieu", key="lieu")
        nom_sugg = f"{date.strftime('%Y-%m-%d')} vs {adversaire}" if adversaire else f"{date.strftime('%Y-%m-%d')}"
        nom_match = st.text_input("Nom du match", value=st.session_state.get("nom_match_sugg", nom_sugg), key="nom_match_sugg")
        formation = st.selectbox("Formation", list(POSTES_DETAILES.keys()), key="match_formation")
        st.session_state["formation_new_match"] = formation
        terrain = terrain_interactif_detaillé(formation, "terrain_new_match")
        tous_titulaires = [v["Nom"] for v in terrain.values() if isinstance(v, dict) and "Nom" in v and v]
        remplaçants = remplaçants_interactif("new_match", tous_titulaires)
        capitaine = st.selectbox("Sélectionner le capitaine", [""] + tous_titulaires, key="capitaine_new_match")
        fig = draw_football_pitch_vertical()
        fig = plot_lineup_on_pitch_vertical(fig, terrain, formation, remplaçants, capitaine)
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key="fig_create_match")
        if st.button("Enregistrer le match"):
            try:
                match_id = nom_match
                if match_id.strip():
                    st.session_state.matches[match_id] = {
                        "type": type_match,
                        "adversaire": adversaire,
                        "date": str(date),
                        "heure": str(heure),
                        "lieu": lieu,
                        "formation": formation,
                        "details": copy.deepcopy(terrain),
                        "remplacants": copy.deepcopy(remplaçants),
                        "capitaine": capitaine,
                        "events": {},
                        "score": "",
                        "score_afc": 0,
                        "score_adv": 0,
                        "noted": False,
                        "homme_du_match": ""
                    }
                    save_all()
                    st.success("Match enregistré !")
                    st.session_state["terrain_new_match"] = {abbr: None for _, abbr in POSTES_DETAILES[formation]}
                    st.session_state["remp_new_match"] = [None] * MAX_REMPLACANTS
                    st.session_state["capitaine_new_match"] = ""
                    st.rerun()
                else:
                    st.warning("Veuillez donner un nom au match.")
            except Exception as e:
                st.error(f"Erreur lors de la sauvegarde : {e}")
                st.text(traceback.format_exc())
    with subtab2:
        if not st.session_state.matches:
            st.info("Aucun match enregistré.")
        else:
            for mid, match in st.session_state.matches.items():
                with st.expander(f"{match['date']} {match['heure']} vs {match['adversaire']} ({match['type']})"):
                    st.write(f"**Statut :** {'Terminé' if match.get('noted', False) else 'En cours'}")
                    st.write(f"**Lieu :** {match['lieu']}")
                    st.write(f"**Formation :** {match['formation']}")
                    fig = draw_football_pitch_vertical()
                    fig = plot_lineup_on_pitch_vertical(fig, match["details"], match["formation"], match.get("remplacants", []), match.get("capitaine"))
                    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"fig_match_{mid}")
                    if match.get("noted", False):
                        score_col1, score_col2, score_col3 = st.columns([2,1,2])
                        with score_col1:
                            st.markdown(f"### {match['adversaire']}")
                        with score_col2:
                            st.markdown(f"### {match.get('score_afc', 0)} - {match.get('score_adv', 0)}")
                        with score_col3:
                            st.markdown("### AFC")
                        st.markdown("---")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown("#### 📊 Stats du match")
                            ev = match.get("events", {})
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
                            st.markdown(f"**🏆 Homme du match :** {match.get('homme_du_match','')}")
                            notes = ev.get("notes", {})
                            if notes:
                                st.markdown("**⭐ Meilleures notes:**")
                                sorted_notes = sorted(notes.items(), key=lambda x: x[1], reverse=True)
                                for nom, note in sorted_notes[:3]:
                                    st.markdown(f"- {nom}: {note}/10")
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
                    else:
                        joueurs_all = [v["Nom"] for v in match["details"].values() if isinstance(v, dict) and "Nom" in v and v]
                        score_afc = st.number_input("Buts AFC", min_value=0, max_value=20, value=match.get("score_afc", 0), key=f"score_afc_{mid}")
                        score_adv = st.number_input(f"Buts {match['adversaire']}", min_value=0, max_value=20, value=match.get("score_adv", 0), key=f"score_adv_{mid}")
                        buteurs_qte = {}
                        st.write("#### Buteurs")
                        for nom in joueurs_all:
                            q = st.number_input(f"{nom} - Buts", min_value=0, max_value=10, value=match.get("events", {}).get("buteurs", {}).get(nom, 0), step=1, key=f"but_{mid}_{nom}")
                            if q > 0:
                                buteurs_qte[nom] = q
                        passeurs_qte = {}
                        st.write("#### Passeurs")
                        for nom in joueurs_all:
                            q = st.number_input(f"{nom} - Passes", min_value=0, max_value=10, value=match.get("events", {}).get("passeurs", {}).get(nom, 0), step=1, key=f"pass_{mid}_{nom}")
                            if q > 0:
                                passeurs_qte[nom] = q
                        cj_qte = {}
                        st.write("#### Cartons jaunes")
                        for nom in joueurs_all:
                            q = st.number_input(f"{nom} - Jaunes", min_value=0, max_value=5, value=match.get("events", {}).get("cartons_jaunes", {}).get(nom, 0), step=1, key=f"cj_{mid}_{nom}")
                            if q > 0:
                                cj_qte[nom] = q
                        cr_qte = {}
                        st.write("#### Cartons rouges")
                        for nom in joueurs_all:
                            q = st.number_input(f"{nom} - Rouges", min_value=0, max_value=2, value=match.get("events", {}).get("cartons_rouges", {}).get(nom, 0), step=1, key=f"cr_{mid}_{nom}")
                            if q > 0:
                                cr_qte[nom] = q
                        notes = {}
                        st.write("#### Notes")
                        for nom in joueurs_all:
                            n = st.number_input(f"{nom} - Note", min_value=0.0, max_value=10.0, value=match.get("events", {}).get("notes", {}).get(nom, 0.0), step=0.1, key=f"note_{mid}_{nom}")
                            if n > 0:
                                notes[nom] = n
                        homme_du_match = st.selectbox("Homme du match", [""] + joueurs_all, key=f"hdm_{mid}")
                        match_ended = st.checkbox("Match terminé", value=match.get("noted", False), key=f"ended_{mid}")
                        if st.button("Valider les stats du match", key=f"valide_{mid}"):
                            match["score"] = f"{score_afc}-{score_adv}"
                            match["score_afc"] = score_afc
                            match["score_adv"] = score_adv
                            match["events"] = {
                                "buteurs": buteurs_qte,
                                "passeurs": passeurs_qte,
                                "cartons_jaunes": cj_qte,
                                "cartons_rouges": cr_qte,
                                "notes": notes
                            }
                            match["noted"] = match_ended
                            match["homme_du_match"] = homme_du_match
                            save_all()
                            st.success("Stats du match enregistrées !")
                            st.rerun()
                    if st.button(f"Supprimer ce match", key=f"suppr_match_{mid}"):
                        del st.session_state.matches[mid]
                        save_all()
                        st.rerun()