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
FORMATION = {
    "4-2-3-1": {"G": 1, "D": 4, "M": 5, "A": 1},
    "4-4-2": {"G": 1, "D": 4, "M": 4, "A": 2},
    "4-3-3": {"G": 1, "D": 4, "M": 3, "A": 3},
    "3-5-2": {"G": 1, "D": 3, "M": 5, "A": 2},
    "3-4-3": {"G": 1, "D": 3, "M": 4, "A": 3},
    "5-3-2": {"G": 1, "D": 5, "M": 3, "A": 2},
}
POSTES_LONG = {"G": "Gardien", "D": "Défenseur", "M": "Milieu", "A": "Attaquant"}
POSTES_ORDER = ["G", "D", "M", "A"]
POSTES_NOMS = {
    "4-2-3-1": {
        "G": ["Gardien"],
        "D": ["Latéral gauche", "Défenseur central gauche", "Défenseur central droit", "Latéral droit"],
        "M": ["Milieu défensif gauche", "Milieu défensif droit", "Ailier gauche", "Milieu offensif", "Ailier droit"],
        "A": ["Avant-centre"]
    },
    "4-3-3": {
        "G": ["Gardien"],
        "D": ["Latéral gauche", "Défenseur central gauche", "Défenseur central droit", "Latéral droit"],
        "M": ["Milieu gauche", "Milieu axial", "Milieu droit"],
        "A": ["Ailier gauche", "Avant-centre", "Ailier droit"]
    },
    "4-4-2": {
        "G": ["Gardien"],
        "D": ["Latéral gauche", "Défenseur central gauche", "Défenseur central droit", "Latéral droit"],
        "M": ["Milieu gauche", "Milieu axial gauche", "Milieu axial droit", "Milieu droit"],
        "A": ["Attaquant gauche", "Attaquant droit"]
    },
    "3-5-2": {
        "G": ["Gardien"],
        "D": ["Défenseur gauche", "Défenseur axial", "Défenseur droit"],
        "M": ["Ailier gauche", "Milieu axial gauche", "Milieu axial", "Milieu axial droit", "Ailier droit"],
        "A": ["Attaquant gauche", "Attaquant droit"]
    },
    "3-4-3": {
        "G": ["Gardien"],
        "D": ["Défenseur gauche", "Défenseur axial", "Défenseur droit"],
        "M": ["Milieu gauche", "Milieu axial gauche", "Milieu axial droit", "Milieu droit"],
        "A": ["Ailier gauche", "Avant-centre", "Ailier droit"]
    },
    "5-3-2": {
        "G": ["Gardien"],
        "D": ["Latéral gauche", "Stoppeur gauche", "Libéro", "Stoppeur droit", "Latéral droit"],
        "M": ["Milieu gauche", "Milieu axial", "Milieu droit"],
        "A": ["Attaquant gauche", "Attaquant droit"]
    }
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
    fig.update_yaxes(showticklabels=False, range=[-25, 125], visible=False)
    fig.update_layout(
        width=460, height=800, plot_bgcolor="#154734", margin=dict(l=10,r=10,t=10,b=10), showlegend=False
    )
    return fig

def positions_for_formation_vertical(formation):
    presets = {
        "4-2-3-1": {
            "G": [(34, 8)],
            "D": [(10, 30), (22, 22), (46, 22), (58, 30)],
            "M": [(18, 40), (50, 40), (10, 60), (34, 60), (58, 60)],
            "A": [(34, 88)],
        },
        "4-3-3": {
            "G": [(34, 8)],
            "D": [(10, 40), (22, 25), (46, 25), (58, 40)],
            "M": [(18, 56), (34, 62.5), (50, 56)],
            "A": [(18, 80), (34, 92), (50, 80)],
        },
        "4-4-2": {
            "G": [(34, 8)],
            "D": [(10, 22), (22, 22), (46, 22), (58, 22)],
            "M": [(10, 48), (22, 48), (46, 48), (58, 48)],
            "A": [(24, 85), (44, 85)],
        },
        "3-5-2": {
            "G": [(34, 8)],
            "D": [(17, 17), (34, 17), (51, 17)],
            "M": [(10, 40), (22, 50), (34, 60), (46, 50), (58, 40)],
            "A": [(24, 88), (44, 88)],
        },
        "3-4-3": {
            "G": [(34, 8)],
            "D": [(17, 17), (34, 17), (51, 17)],
            "M": [(10, 46), (22, 56), (46, 56), (58, 46)],
            "A": [(18, 80), (34, 92), (50, 80)],
        },
        "5-3-2": {
            "G": [(34, 8)],
            "D": [(7, 20), (18, 20), (34, 20), (50, 20), (61, 20)],
            "M": [(15, 52.5), (34, 60), (53, 52.5)],
            "A": [(24, 88), (44, 88)],
        },
    }
    return presets.get(formation, presets["4-2-3-1"])

def plot_lineup_on_pitch_vertical(fig, details, formation, remplacants=None, player_stats=None):
    positions = positions_for_formation_vertical(formation)
    color_poste = "#0d47a1"
    for poste in POSTES_ORDER:
        for i, joueur in enumerate(details.get(poste, [])):
            if joueur and isinstance(joueur, dict) and "Nom" in joueur:
                x, y = positions[poste][i % len(positions[poste])]
                nom = joueur["Nom"]
                # Gather stats
                stats = ""
                if player_stats and nom in player_stats:
                    s = player_stats[nom]
                    parts = []
                    if s.get("buts"):
                        parts.append(f"⚽ {s['buts']}")
                    if s.get("passes"):
                        parts.append(f"🎯 {s['passes']}")
                    if s.get("cj"):
                        parts.append(f"🟨 {s['cj']}")
                    if s.get("cr"):
                        parts.append(f"🟥 {s['cr']}")
                    if s.get("note"):
                        parts.append(f"⭐ {s['note']}")
                    if s.get("hdm"):
                        parts.append("🏆")
                    stats = " | ".join(parts)
                # Hovertext with stats
                hovertext = f"{nom}{' (C)' if joueur.get('Capitaine') else ''}"
                if stats:
                    hovertext += f"<br/>{stats}"
                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode="markers+text",
                    marker=dict(size=38, color=color_poste, line=dict(width=2, color="white")),
                    text=f"{joueur.get('Numero', '')}".strip() if "Numero" in joueur else "",
                    textposition="middle center",
                    textfont=dict(color="white", size=17, family="Arial Black"),
                    hovertext=hovertext,
                    hoverinfo="text"
                ))
                # Show stats as subtitle below player
                if stats:
                    fig.add_trace(go.Scatter(
                        x=[x], y=[y-9],
                        mode="text",
                        text=[stats],
                        textfont=dict(color="yellow", size=12, family="Arial Black"),
                        showlegend=False
                    ))
                fig.add_trace(go.Scatter(
                    x=[x], y=[y-6],
                    mode="text",
                    text=[nom + (" (C)" if joueur.get("Capitaine") else "")],
                    textfont=dict(color="white", size=13, family="Arial Black"),
                    showlegend=False
                ))
    remplacants = remplacants or []
    n = len(remplacants)
    if n:
        # Définition des positions pour 2 lignes : 3 en haut, 2 en bas (max 5 remplaçants)
        positions = []
        if n == 1:
            positions = [(34, -10)]
        elif n == 2:
            positions = [(28, -10), (40, -10)]
        elif n == 3:
            positions = [(22, -10), (34, -10), (46, -10)]
        elif n == 4:
            positions = [(22, -8), (34, -8), (46, -8), (34, -17)]
        else:
            # Cas général (n>=5) : 3 en haut, 2 en bas
            positions = [(18, -8), (34, -8), (50, -8), (26, -17), (42, -17)]
        for idx, remp in enumerate(remplacants):
            if idx >= len(positions):
                break  # N'afficher que 5 remplaçants max
            x_r, y_r = positions[idx]
            if isinstance(remp, dict):
                nom = remp.get("Nom", "")
                numero = remp.get("Numero", "")
            else:
                nom = remp
                numero = ""
            fig.add_trace(go.Scatter(
                x=[x_r], y=[y_r],
                mode="markers+text",
                marker=dict(size=28, color="#0d47a1", line=dict(width=2, color="white")),
                text=[str(numero)],
                textposition="middle center",
                textfont=dict(color="white", size=16, family="Arial Black"),
                hovertext=[str(nom)],
                hoverinfo="text"
            ))
            fig.add_trace(go.Scatter(
                x=[x_r], y=[y_r-5],
                mode="text",
                text=[f"{nom}"],
                textfont=dict(color="white", size=12, family="Arial Black"),
                showlegend=False
            ))
    return fig

def save_all():
    data = {
        "players": st.session_state.players.to_dict(orient="records"),
        "lineups": st.session_state.lineups,
        "matches": st.session_state.matches,
    }
    try:
        with open(DATA_FILE, "w") as f:
            json.dump(data, f, indent=2)
        print("Données sauvegardées dans le fichier JSON !")
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        st.session_state.players = pd.DataFrame(data.get("players", []))
        st.session_state.lineups = data.get("lineups", {})
        st.session_state.matches = data.get("matches", {})
    except Exception as e:
        st.error(f"Erreur lors de la sauvegarde du fichier JSON : {e}")
        st.text(traceback.format_exc())

def fusion_dictionnaires(json_dict, session_dict):
    fusion = dict(json_dict)
    fusion.update(session_dict)
    return fusion

def reload_all():
    if os.path.exists(DATA_FILE):
        with open(DATA_FILE, "r") as f:
            data = json.load(f)
        session_players = st.session_state.get("players", pd.DataFrame(columns=PLAYER_COLS))
        json_players = pd.DataFrame(data.get("players", []))
        if not session_players.empty:
            merged_players = pd.concat([json_players, session_players]).drop_duplicates(subset="Nom", keep="last")
        else:
            merged_players = json_players
        st.session_state.players = merged_players

        session_lineups = st.session_state.get("lineups", {})
        json_lineups = data.get("lineups", {})
        st.session_state.lineups = fusion_dictionnaires(json_lineups, session_lineups)

        session_matches = st.session_state.get("matches", {})
        json_matches = data.get("matches", {})
        st.session_state.matches = fusion_dictionnaires(json_matches, session_matches)
    else:
        st.session_state.players = pd.DataFrame(columns=PLAYER_COLS)
        st.session_state.lineups = {}
        st.session_state.matches = {}

def terrain_init(formation):
    return {poste: [None for _ in range(FORMATION[formation][poste])] for poste in POSTES_ORDER}

def terrain_interactif(formation, terrain_key):
    if terrain_key not in st.session_state:
        st.session_state[terrain_key] = {poste: [None for _ in range(FORMATION[formation][poste])] for poste in POSTES_ORDER}
    terrain = st.session_state[terrain_key]
    # Affichage vertical par poste, compatible mobile
    for poste in POSTES_ORDER:
        if formation in POSTES_NOMS and poste in POSTES_NOMS[formation]:
            noms_postes = POSTES_NOMS[formation][poste]
        else:
            noms_postes = [f"{POSTES_LONG[poste]} {i+1}" for i in range(FORMATION[formation][poste])]
        for i in range(FORMATION[formation][poste]):
            all_selected = [j["Nom"] for p in POSTES_ORDER for j in terrain.get(p, []) if isinstance(j, dict) and "Nom" in j and j]
            current_joueur = terrain[poste][i] if (isinstance(terrain[poste][i], dict) and terrain[poste][i] and "Nom" in terrain[poste][i]) else None
            current_nom = current_joueur["Nom"] if current_joueur else ""
            label = noms_postes[i] if i < len(noms_postes) else f"{POSTES_LONG[poste]} {i+1}"
            joueur_options = [""] + [
                n for n in st.session_state.players["Nom"]
                if n and (n == current_nom or n not in all_selected)
            ]
            choix = st.selectbox(
                label,
                joueur_options,
                index=joueur_options.index(current_nom) if current_nom in joueur_options else 0,
                key=f"{terrain_key}_{poste}_{i}"
            )
            if choix:
                joueur_info = st.session_state.players[st.session_state.players["Nom"] == choix].iloc[0].to_dict()
                num = st.text_input(f"Numéro de {choix}", value=current_joueur.get("Numero","") if current_joueur else "", key=f"num_{terrain_key}_{poste}_{i}")
                cap = st.checkbox(f"Capitaine ?", value=current_joueur.get("Capitaine", False) if current_joueur else False, key=f"cap_{terrain_key}_{poste}_{i}")
                joueur_info["Numero"] = num
                joueur_info["Capitaine"] = cap
                terrain[poste][i] = joueur_info
            else:
                terrain[poste][i] = None
    st.session_state[terrain_key] = terrain
    return terrain
    
def remplacants_interactif(key, titulaires):
    if f"remp_{key}" not in st.session_state:
        st.session_state[f"remp_{key}"] = [{"Nom": None, "Numero": ""} for _ in range(MAX_REMPLACANTS)]
    remps = st.session_state[f"remp_{key}"]

    # Patch robustesse colonne Nom
    if hasattr(st.session_state.players, "columns") and "Nom" in st.session_state.players.columns:
        noms_joueurs = st.session_state.players["Nom"].dropna().astype(str).tolist()
    else:
        noms_joueurs = []
    dispo = [n for n in noms_joueurs if n not in titulaires and n not in [r["Nom"] for r in remps if r["Nom"]] if n]
    for i in range(MAX_REMPLACANTS):
        current = remps[i]["Nom"]
        options = dispo + ([current] if current and current not in dispo else [])
        choix = st.selectbox(
            f"Remplacant {i+1}",
            [""] + options,
            index=(options.index(current)+1) if current in options else 0,
            key=f"remp_choice_{key}_{i}"
        )
        if choix:
            joueur_info = st.session_state.players[st.session_state.players["Nom"] == choix].iloc[0].to_dict()
            num = st.text_input(f"Numéro de {choix}", value=remps[i].get("Numero",""), key=f"num_remp_{key}_{i}")
            remps[i] = {"Nom": choix, "Numero": num}
        else:
            remps[i] = {"Nom": None, "Numero": ""}
        dispo = [n for n in dispo if n != choix]
    st.session_state[f"remp_{key}"] = remps
    # On renvoie la liste filtrée des remplacants valides
    return [r for r in remps if r["Nom"]]

def compute_player_stats(joueur_nom):
    buts = passes = cj = cr = selections = titularisations = note_sum = note_count = hdm = 0
    for match in st.session_state.matches.values():
        details = match.get("details", {})
        joueurs = [j for p in POSTES_ORDER for j in details.get(p, []) if j and isinstance(j, dict) and j.get("Nom") == joueur_nom]
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

def compute_clean_sheets():
    # Returns a dict: {player_name: clean_sheet_count}
    if "matches" not in st.session_state:
        return {}
    clean_sheets = {}
    for match in st.session_state.matches.values():
        # Only count if match is finished
        if not match.get("noted", False):
            continue
        score_adv = match.get("score_adv", 0)
        if score_adv != 0:
            continue
        # Find the goalkeeper(s) in the match details
        details = match.get("details", {})
        for joueur in details.get("G", []):
            if joueur and isinstance(joueur, dict) and joueur.get("Nom"):
                name = joueur["Nom"]
                clean_sheets[name] = clean_sheets.get(name, 0) + 1
    return clean_sheets

if ("players" not in st.session_state or
    "lineups" not in st.session_state or
    "matches" not in st.session_state):
    reload_all()
if not isinstance(st.session_state.players, pd.DataFrame):
    st.session_state.players = pd.DataFrame(columns=PLAYER_COLS)
for col in PLAYER_COLS:
    if col not in st.session_state.players.columns:
        st.session_state.players[col] = ""
if "formation" not in st.session_state:
    st.session_state.formation = DEFAULT_FORMATION

def download_upload_buttons():
    # -- Import JSON --
    with st.form("import_json_form"):
        up_json = st.file_uploader("📂 Importer un fichier JSON", type="json", key="upload_all")
        submitted = st.form_submit_button("📤 Importer ce fichier")
        if submitted and up_json:
            try:
                data = json.load(up_json)
                st.session_state.players = pd.DataFrame(data.get("players", []))
                st.session_state.lineups = data.get("lineups", {})
                st.session_state.matches = data.get("matches", {})
                st.success("✅ Données importées dans la session. N'oubliez pas de cliquer sur les boutons Sauvegarder dans les menus pour valider sur disque.")
            except Exception as e:
                st.error(f"❌ Erreur à l'import : {e}")

    # -- Export JSON (depuis la session courante) --
    st.download_button(
        label="💾 Télécharger le fichier JSON (état courant)",
        data=json.dumps({
            "players": st.session_state.players.to_dict(orient="records"),
            "lineups": st.session_state.lineups,
            "matches": st.session_state.matches,
        }, indent=2),
        file_name=DATA_FILE,
        mime="application/json"
    )
    

st.sidebar.title("⚽ Gestion Équipe AFC")
with st.sidebar:
    st.markdown("---")
    with st.expander("🔄 Import/Export des données"):
        download_upload_buttons()
    st.markdown("---")

tab3, tab4, tab1, tab2 = st.tabs(["Matchs", "Stats équipe", "Base joueurs", "Compositions"])

# --- DATABASE ---
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
        st.session_state.players = edited_df[PLAYER_COLS]
        save_all()
        st.rerun()
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
            del st.session_state[edit_key]
        nom_compo = st.text_input("Nom de la composition", key="nom_compo_create", value=nom_compo if edit_compo else "")
        formation = st.selectbox(
            "Formation", list(FORMATION.keys()),
            index=list(FORMATION.keys()).index(st.session_state.get("formation_create_compo", DEFAULT_FORMATION)),
            key="formation_create_compo"
        )
        terrain = terrain_interactif(formation, "terrain_create_compo")
        tous_titulaires = [j["Nom"] for p in POSTES_ORDER for j in terrain.get(p, []) if j and isinstance(j, dict) and "Nom" in j]
        remplacants = remplacants_interactif("create_compo", tous_titulaires)
        fig = draw_football_pitch_vertical()
        fig = plot_lineup_on_pitch_vertical(fig, terrain, formation, remplacants)
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key="fig_create_compo")
        if st.button("Sauvegarder la composition"):
            if not nom_compo.strip():
                st.error("Merci d'indiquer un nom pour la composition.")
                st.stop()
            try:
                lineup = {
                    "formation": formation,
                    "details": copy.deepcopy(terrain),
                    "remplacants": copy.deepcopy(remplacants)
                }
                st.session_state.lineups[nom_compo] = lineup
                save_all()
                st.rerun()
                st.success("Composition sauvegardée !")
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
                    fig = plot_lineup_on_pitch_vertical(fig, compo["details"], compo["formation"], compo.get("remplacants", []))
                    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"fig_compo_{nom}")
                    col1, col2 = st.columns(2)
                    if col1.button(f"Éditer {nom}", key=f"edit_{nom}"):
                        st.session_state["edit_compo"] = (nom, compo)
                        st.rerun()
                    if col2.button(f"Supprimer {nom}", key=f"suppr_{nom}"):
                        del st.session_state.lineups[nom]
                        save_all()
                        st.rerun()
                        st.success("Composition supprimée !")

# --- MATCHS ---
with tab3:
    st.title("Gestion des matchs")
    subtab1, subtab2 = st.tabs(["Créer un match", "Mes matchs"])
    with subtab1:
        if st.button("Réinitialiser la création du match"):
            for k in [
                "terrain_new_match", "formation_new_match",
                "remp_new_match", "nom_match_sugg", "adversaire", "lieu"
            ]:
                if k in st.session_state:
                    del st.session_state[k]
            st.rerun()
        type_match = st.selectbox("Type de match", ["Championnat", "Coupe"], key="type_match")
        if type_match=="Championnat":
            journee= st.text_input("Journée", value="J", key="journee")
        else:
            journee= st.selectbox("Tour", ["Poules", "Huitièmes", "Quarts", "Demies", "Finale"], key="journee")
        adversaire = st.text_input("Nom de l'adversaire", key="adversaire")
        date = st.date_input("Date du match", value=datetime.today())
        heure = st.time_input("Heure du match", value="21:00")
        domicile = st.selectbox("Domicile/Extérieur", ["Domicile", "Extérieur"])
        if domicile == "Domicile":
            lieu = st.text_input("Lieu", value="Club de Football Barradels, 2 Rue des Cyclamens, 31700 Blagnac", key="lieu")
            nom_sugg = f"{type_match} - {journee} - AFC vs {adversaire}" if adversaire else f"{type_match} - {journee}"
        else:
            lieu = st.text_input("Lieu", key="lieu")
            nom_sugg = f"{type_match} - {journee} - {adversaire} vs AFC" if adversaire else f"{type_match} - {journee}"
        nom_match = st.text_input("Nom du match", value=st.session_state.get("nom_match_sugg", nom_sugg), key="nom_match_sugg")
        use_compo = st.checkbox("Utiliser une composition enregistrée ?", key="use_compo_match")
        if use_compo and st.session_state.lineups:
            compo_keys = list(st.session_state.lineups.keys())
            # Initialisation de la sélection si besoin
            if "compo_choice_match" not in st.session_state:
                st.session_state.compo_choice_match = compo_keys[0] if compo_keys else ""
            compo_choice = st.selectbox(
                "Choisir la composition",
                compo_keys,
                index=compo_keys.index(st.session_state.get("compo_choice_match", compo_keys[0])) if compo_keys else 0,
                key="compo_choice_match"
            )
            compo_data = st.session_state.lineups[compo_choice]
            formation = compo_data["formation"]
            terrain = copy.deepcopy(compo_data["details"])
            remplacants = list(compo_data.get("remplacants", []))
            st.session_state["formation_new_match"] = formation
            st.session_state["terrain_new_match"] = terrain
            st.session_state["remp_new_match"] = remplacants
        else:
            formation = st.selectbox("Formation", list(FORMATION.keys()), key="match_formation")
            st.session_state["formation_new_match"] = formation
            terrain = terrain_interactif(formation, "terrain_new_match")
            tous_titulaires = [j["Nom"] for p in POSTES_ORDER for j in terrain.get(p, []) if j and isinstance(j, dict) and "Nom" in j]
            remplacants = remplacants_interactif("new_match", tous_titulaires)
        fig = draw_football_pitch_vertical()
        fig = plot_lineup_on_pitch_vertical(fig, terrain, formation, remplacants)
        st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key="fig_create_match")
        if st.button("Enregistrer le match"):
            try:
                match_id = nom_match
                st.session_state.matches[match_id] = {
                    "type": type_match,
                    "adversaire": adversaire,
                    "date": str(date),
                    "heure": str(heure),
                    "domicile" : domicile, 
                    "journée" : journee, 
                    "nom_match" : nom_match, 
                    "lieu": lieu,
                    "formation": formation,
                    "details": copy.deepcopy(terrain),
                    "remplacants": copy.deepcopy(remplacants),
                    "events": {},
                    "score": "",
                    "score_afc": 0,
                    "score_adv": 0,
                    "noted": False,
                    "homme_du_match": ""
                }
                save_all()
                st.rerun()
                st.success("Match enregistré !")
            except Exception as e:
                import traceback
                st.error(f"Erreur lors de la sauvegarde : {e}")
                st.text(traceback.format_exc())
    with subtab2:
        if not st.session_state.matches:
            st.info("Aucun match enregistré.")
        else:
            for mid, match in st.session_state.matches.items():
                with st.expander(st.session_state.get("nom_match_sugg", nom_match)):
                    st.write(f"**Statut :** {'Terminé' if match.get('noted', False) else 'A jouer'}")
                    if match.get("noted", False):
                        score_col1, score_col2, score_col3 = st.columns([2,1,2])
                        if match.get('domicile')=="Domicile":
                            with score_col1:
                                st.markdown("### AFC")
                            with score_col2:
                                st.markdown(f"### {match.get('score_afc', 0)} - {match.get('score_adv', 0)}")
                            with score_col3:
                                st.markdown(f"### {match['adversaire']}")
                        else:
                            with score_col1:
                                st.markdown(f"### {match['adversaire']}")
                            with score_col2:
                                st.markdown(f"{match.get('score_adv', 0)} - ### {match.get('score_afc', 0)}")
                            with score_col3:
                                st.markdown("### AFC")
                    st.write(f"{match['nom_match']}")
                    st.markdown("---")
                    fig = draw_football_pitch_vertical()
                    # Prepare player stats for display
                    ev = match.get("events", {})
                    joueurs_all = [j['Nom'] for p in POSTES_ORDER for j in match["details"].get(p, []) if j and isinstance(j, dict) and "Nom" in j]
                    player_stats = {}
                    for nom in joueurs_all:
                        player_stats[nom] = {
                            "buts": ev.get("buteurs", {}).get(nom, 0),
                            "passes": ev.get("passeurs", {}).get(nom, 0),
                            "cj": ev.get("cartons_jaunes", {}).get(nom, 0),
                            "cr": ev.get("cartons_rouges", {}).get(nom, 0),
                            "note": ev.get("notes", {}).get(nom, None),
                            "hdm": match.get("homme_du_match", "") == nom
                        }
                    fig = plot_lineup_on_pitch_vertical(
                        fig,
                        match["details"],
                        match["formation"],
                        match.get("remplacants", []),
                        player_stats=player_stats
                    )
                    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"fig_match_{mid}")
                    st.markdown("---")
                    if match.get("noted", False):    
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
                        st.session_state[f"formation_terrain_match_{mid}"] = match["formation"]
                        st.session_state[f"terrain_match_{mid}"] = match["details"]
                        terrain = terrain_interactif(match["formation"], f"terrain_match_{mid}")
                        remp_edit = remplacants_interactif(f"edit_match_{mid}", [j["Nom"] for p in POSTES_ORDER for j in match["details"].get(p, []) if j and isinstance(j, dict) and "Nom" in j])
                        if st.button("Mettre à jour la compo", key=f"maj_compo_{mid}"):
                            match["details"] = st.session_state.get(f"terrain_match_{mid}", match["details"])
                            match["remplacants"] = remp_edit
                            save_all()
                            st.success("Composition du match mise à jour.")
                            st.rerun()
                    match_ended = st.checkbox("Match terminé", value=match.get("noted", False), key=f"ended_{mid}")
                    if match_ended and not match.get("noted", False):
                        st.write("### Saisie des stats du match")
                        titularies = [j['Nom'] for p in POSTES_ORDER for j in match["details"].get(p, []) if j and isinstance(j, dict) and "Nom" in j]
                        remplaçants = [r["Nom"] for r in match.get("remplacants", []) if isinstance(r, dict) and r.get("Nom")]
                        joueurs_all = list(dict.fromkeys(titularies + remplaçants))  # Keeps order, removes duplicates
                        score_afc = st.number_input("Buts AFC", min_value=0, max_value=20, value=0, key=f"score_afc_{mid}")
                        score_adv = st.number_input(f"Buts {match['adversaire']}", min_value=0, max_value=20, value=0, key=f"score_adv_{mid}")
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
                            q = st.number_input(f"{nom} - Jaunes", min_value=0, max_value=5, value=0, step=1, key=f"cj_{mid}_{nom}")
                            if q > 0:
                                cj_qte[nom] = q
                        cr_qte = {}
                        st.write("#### Cartons rouges")
                        for nom in joueurs_all:
                            q = st.number_input(f"{nom} - Rouges", min_value=0, max_value=2, value=0, step=1, key=f"cr_{mid}_{nom}")
                            if q > 0:
                                cr_qte[nom] = q
                        notes = {}
                        st.write("#### Notes")
                        for nom in joueurs_all:
                            n = st.number_input(f"{nom} - Note", min_value=0.0, max_value=10.0, value=0.0, step=0.1, key=f"note_{mid}_{nom}")
                            if n > 0:
                                notes[nom] = n
                        homme_du_match = st.selectbox("Homme du match", [""] + joueurs_all, key=f"hdm_{mid}")
                        if st.button("Valider le match", key=f"valide_{mid}"):
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
                            match["noted"] = True
                            match["homme_du_match"] = homme_du_match
                            save_all()
                            st.success("Stats du match enregistrées !")
                            st.rerun()
                    if st.button(f"Supprimer ce match", key=f"suppr_match_{mid}"):
                        del st.session_state.matches[mid]
                        save_all()
                        st.rerun()

#----STATS EQUIPE-----
with tab4:
    st.title("📊 Statistiques de l'équipe")
    stats_data = []
    for _, row in st.session_state.players.iterrows():
        s = compute_player_stats(row["Nom"])
        stats_data.append({**row, **s})

    df = pd.DataFrame(stats_data)
    clean_sheets = compute_clean_sheets()
    if not df.empty:
        df["Clean sheets"] = df.apply(
            lambda r: clean_sheets.get(r["Nom"], 0) if r["Poste"] == "G" else None, axis=1)
        df["Bouchers"] = df["Cartons rouges"].fillna(0) + df["Cartons jaunes"].fillna(0)
    
        # Top 5 by rating
        top_rating = df[df["Note générale"] > 0].sort_values("Note générale", ascending=False).head(5)
        # Top 5 scorers
        top_buts = df[df["Buts"] > 0].sort_values("Buts", ascending=False).head(5)
        # Top 5 passers
        top_passes = df[df["Passes décisives"] > 0].sort_values("Passes décisives", ascending=False).head(5)
        # Top 5 decisive
        top_decisive = df[df["Buts + Passes"] > 0].sort_values("Buts + Passes", ascending=False).head(5)
        # Top 5 clean sheets (goalkeepers only)
        top_clean = df[df["Poste"] == "G"].sort_values("Clean sheets", ascending=False).head(5)
        # Top 5 ratio
        top_ratio = df[df["Décisif par match"] > 0].sort_values("Décisif par match", ascending=False).head(5)
        # Top 5 used
        top_used = df[df["Titularisations"] > 0].sort_values("Titularisations", ascending=False).head(5)
        # Top 5 bouchers (by red, then yellow)
        top_bouchers = df[(df["Cartons rouges"] > 0) | (df["Cartons jaunes"] > 0)].sort_values(
            by=["Cartons rouges", "Cartons jaunes"], ascending=[False, False]).head(5)
    
        col1, col2 = st.columns(2)
    
        with col1:
            st.subheader("⭐ Top 5 Notes")
            st.dataframe(top_rating[["Nom", "Note générale"]], use_container_width=True)
            st.subheader("⚽ Top 5 Buteurs")
            st.dataframe(top_buts[["Nom", "Buts"]], use_container_width=True)
            st.subheader("🎯 Top 5 Passeurs")
            st.dataframe(top_passes[["Nom", "Passes décisives"]], use_container_width=True)
            st.subheader("🔥 Top 5 Décisifs (Buts+Passes)")
            st.dataframe(top_decisive[["Nom", "Buts + Passes"]], use_container_width=True)
            st.subheader("🧤 Top 5 Clean Sheets (Gardiens)")
            st.dataframe(top_clean[["Nom", "Clean sheets"]], use_container_width=True)
    
        with col2:
            st.subheader("⚡ Top 5 Ratio Décisif/Match")
            st.dataframe(top_ratio[["Nom", "Décisif par match"]], use_container_width=True)
            st.subheader("🔁 Top 5 Plus Utilisés")
            st.dataframe(top_used[["Nom", "Titularisations"]], use_container_width=True)
            st.subheader("🟥🟨 Top 5 Bouchers")
            st.dataframe(top_bouchers[["Nom", "Cartons rouges", "Cartons jaunes"]], use_container_width=True)
    
        # Team stats
        total_goals = df["Buts"].sum()
        total_conceded = sum(
            match.get("score_adv", 0)
            for match in st.session_state.matches.values()
            if match.get("noted", False)
        )
        diff_scorers = df[df["Buts"] > 0]["Nom"].nunique()
    
        st.markdown("---")
        col3, col4, col5 = st.columns(3)
        with col3:
            st.metric("Buts marqués", int(total_goals))
        with col4:
            st.metric("Buts encaissés", int(total_conceded))
        with col5:
            st.metric("Nombre de buteurs différents", int(diff_scorers))
    else:
        st.info("Aucun joueur dans la base, impossible de calculer les statistiques.")
