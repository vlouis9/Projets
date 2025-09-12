# --- 📦 IMPORTS ---
import streamlit as st
import pandas as pd
import json
import base64
import requests
import uuid
from datetime import datetime, timedelta
import plotly.graph_objects as go

# --- 📁 GESTIONNAIRE DE DONNÉES ---
class AFCDataManager:
    def __init__(self, token=None):
        self.token = token or st.secrets["github"]["token"]
        self.username = "vlouis9"
        self.repo = "Projets"
        self.branch = "main"
        self.file_path = "afcdata.json"
        self.api_url = f"https://api.github.com/repos/{self.username}/{self.repo}/contents/{self.file_path}"
        self.headers = {
            "Authorization": f"token {self.token}",
            "Accept": "application/vnd.github+json"
        }

    def load(self):
        try:
            resp = requests.get(self.api_url, headers=self.headers)
            resp.raise_for_status()
            raw = resp.json()
            content = base64.b64decode(raw["content"]).decode()
            data = json.loads(content)

            st.session_state.players = pd.DataFrame(data.get("players", []))
            st.session_state.lineups = data.get("lineups", {})
            st.session_state.matchs = data.get("matchs", {})
            st.session_state.adversaires = data.get("adversaires", [])
            st.session_state.championnat_scores = data.get("championnat_scores", {})
            st.session_state.profondeur_effectif = data.get("profondeur_effectif", {})
        except Exception as e:
            st.error(f"❌ Échec du chargement des données : {e}")

    def save(self):
        try:
            payload = {
                "players": st.session_state.players.to_dict(orient="records"),
                "lineups": st.session_state.lineups,
                "matchs": st.session_state.matchs,
                "adversaires": st.session_state.adversaires,
                "championnat_scores": st.session_state.championnat_scores,
                "profondeur_effectif": st.session_state.profondeur_effectif
            }
            encoded = base64.b64encode(json.dumps(payload, indent=2).encode()).decode()

            get_resp = requests.get(self.api_url, headers=self.headers)
            sha = get_resp.json()["sha"]

            update = {
                "message": "🔄 Mise à jour via AFCDataManager",
                "content": encoded,
                "branch": self.branch,
                "sha": sha
            }

            put_resp = requests.put(self.api_url, headers=self.headers, json=update)
            if put_resp.status_code in [200, 201]:
                st.success("✅ Données sauvegardées sur GitHub")
            else:
                st.error(f"❌ Erreur GitHub : {put_resp.status_code}")
        except Exception as e:
            st.error(f"❌ Échec de la sauvegarde : {e}")

# --- ⚙️ INIT SESSION + CHARGEMENT DONNÉES ---
manager = AFCDataManager()
manager.load()

# 🧩 Initialiser structures vides si besoin
if not isinstance(st.session_state.players, pd.DataFrame):
    st.session_state.players = pd.DataFrame(columns=["Nom", "Poste", "Infos"])
for col in ["Nom", "Poste", "Infos"]:
    if col not in st.session_state.players.columns:
        st.session_state.players[col] = ""

for key in ["lineups", "matchs", "adversaires", "championnat_scores", "profondeur_effectif"]:
    if key not in st.session_state:
        st.session_state[key] = {}

# --- ⚙️ CONSTANTES FORMATION ---
PLAYER_COLS = ["Nom", "Poste", "Infos"]
DEFAULT_FORMATION = "4-2-3-1"
MAX_REMPLACANTS = 5

FORMATION = {
    "4-2-3-1": {"G": 1, "D": 4, "M": 5, "A": 1},
    "4-4-2":   {"G": 1, "D": 4, "M": 4, "A": 2},
    "4-3-3":   {"G": 1, "D": 4, "M": 3, "A": 3},
    "3-5-2":   {"G": 1, "D": 3, "M": 5, "A": 2},
    "3-4-3":   {"G": 1, "D": 3, "M": 4, "A": 3},
    "5-3-2":   {"G": 1, "D": 5, "M": 3, "A": 2},
}

POSTES_ORDER = ["G", "D", "M", "A"]
POSTES_LONG = {"G": "Gardien", "D": "Défenseur", "M": "Milieu", "A": "Attaquant"}
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

# --- 🎯 POSITIONNEMENT & TERRAIN ---

def draw_football_pitch_vertical():
    fig = go.Figure()

    # 📐 Terrain principal
    fig.add_shape(type="rect", x0=0, y0=0, x1=68, y1=105, line=dict(width=2, color="#145A32"))

    # 🧤 Surface de réparation
    fig.add_shape(type="rect", x0=13.84, y0=0, x1=54.16, y1=16.5, line=dict(width=1, color="#145A32"))
    fig.add_shape(type="rect", x0=13.84, y0=88.5, x1=54.16, y1=105, line=dict(width=1, color="#145A32"))

    # ⚽ Centre du terrain
    fig.add_shape(type="circle", x0=24.85, y0=43.35, x1=43.15, y1=61.65, line=dict(width=1, color="#145A32"))
    fig.add_shape(type="circle", x0=33.6, y0=52.1, x1=34.4, y1=52.9, fillcolor="#145A32", line=dict(color="#145A32"))

    fig.update_xaxes(showticklabels=False, range=[-5, 73], visible=False)
    fig.update_yaxes(showticklabels=False, range=[-35, 125], visible=False)

    fig.update_layout(
        width=460,
        height=1000,
        plot_bgcolor="#154734",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False,
        shapes=[dict(layer="below")]
    )

    return fig

# --- 📍 Positions des joueurs selon formation ---
def positions_for_formation_vertical(formation):
    presets = {
        "4-2-3-1": {
            "G": [(34, 8)],
            "D": [(10, 35), (22, 22), (46, 22), (58, 35)],
            "M": [(18, 50), (50, 50), (10, 70), (34, 70), (58, 70)],
            "A": [(34, 95)],
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

# --- 🧩 Ajout des joueurs & stats sur le terrain ---
def plot_lineup_on_pitch_vertical(fig, details, formation, remplacants=None, player_stats=None):
    positions = positions_for_formation_vertical(formation)
    color_poste = "#0d47a1"
    # Capitaine : recherche le nom
    nom_capitaine = None
    for poste in POSTES_ORDER:
        for joueur in details.get(poste, []):
            if joueur and isinstance(joueur, dict) and joueur.get("Capitaine", False):
                nom_capitaine = joueur["Nom"]

    for poste in POSTES_ORDER:
        for i, joueur in enumerate(details.get(poste, [])):
            if joueur and isinstance(joueur, dict) and "Nom" in joueur:
                x, y = positions[poste][i % len(positions[poste])]
                nom = joueur["Nom"]
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
                    # Affiche note seulement si elle existe (option "noter" cochée)
                    if s.get("note") is not None:
                        parts.append(f"⭐ {s['note']}")
                    if s.get("hdm"):
                        parts.append("🏆")
                    stats = " | ".join(parts)
                # Affichage du capitaine (C) à côté du nom
                hovertext = f"{nom}{' (C)' if joueur.get('Capitaine') or (nom_capitaine and nom == nom_capitaine) else ''}"
                if stats:
                    hovertext += f"<br/>{stats}"

                fig.add_trace(go.Scatter(
                    x=[x], y=[y],
                    mode="markers+text",
                    marker=dict(size=38, color=color_poste, line=dict(width=2, color="white")),
                    text=f"{joueur.get('Numero', '')}".strip(),
                    textposition="middle center",
                    textfont=dict(color="white", size=17, family="Arial Black"),
                    hovertext=hovertext,
                    hoverinfo="text"
                ))
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
                    text=[nom + (" (C)" if joueur.get("Capitaine") or (nom_capitaine and nom == nom_capitaine) else "")],
                    textfont=dict(color="white", size=13, family="Arial Black"),
                    showlegend=False
                ))

    # Remplaçants : positions variables selon nombre
    remplacants = remplacants or []
    n = len(remplacants)
    if n:
        positions_remp = []
        # Répartition plus flexible
        if n<4:
            for i in range(n):
                x = 5 + int((i / max(n-1, 1)) * 58)
                y = -10
                positions_remp.append((x, y))
        else:
            n_top = (n + 1) // 2 
            n_bottom = n - n_top
            for i in range(n_top):
                x = 5 + int((i / max(n_top-1, 1)) * 58)
                y = -10
                positions_remp.append((x, y))
            for i in range(n_bottom):
                x = 5 + int((i / max(n_bottom-1, 1)) * 58)
                y = -20
                positions_remp.append((x, y))
                                      
        for idx, remp in enumerate(remplacants):
            x_r, y_r = positions_remp[idx]
            nom = remp.get("Nom", "") if isinstance(remp, dict) else remp
            numero = remp.get("Numero", "") if isinstance(remp, dict) else ""
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
                if s.get("note") is not None:
                    parts.append(f"⭐ {s['note']}")
                if s.get("hdm"):
                    parts.append("🏆")
                stats = " | ".join(parts)
            if stats:
                fig.add_trace(go.Scatter(
                    x=[x_r], y=[y_r - 9],
                    mode="text",
                    text=[stats],
                    textfont=dict(color="yellow", size=12, family="Arial Black"),
                    showlegend=False
                ))
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

# --- 📊 STATISTIQUES JOUEURS ---

def compute_player_stats(joueur_nom):
    buts = passes = cj = cr = selections = titularisations = note_sum = note_count = hdm = 0
    matchs = st.session_state.get("matchs", {})
    for match in matchs.values():
        if not match.get("termine") and not match.get("noted"):
            continue
        details = match.get("details", {})
        joueurs = [
            j for p in POSTES_ORDER
            for j in details.get(p, [])
            if j and isinstance(j, dict) and j.get("Nom") == joueur_nom
        ]
        if joueurs or joueur_nom in [r.get("Nom") for r in match.get("remplacants", []) if isinstance(r, dict)]:
            selections += 1
        if joueurs:
            titularisations += 1
        events = match.get("events", {})
        buts += events.get("buteurs", {}).get(joueur_nom, 0)
        passes += events.get("passeurs", {}).get(joueur_nom, 0)
        cj += events.get("cartons_jaunes", {}).get(joueur_nom, 0)
        cr += events.get("cartons_rouges", {}).get(joueur_nom, 0)
        if joueur_nom in events.get("notes", {}):
            note_sum += events["notes"][joueur_nom]
            note_count += 1
        if match.get("homme_du_match") == joueur_nom:
            hdm += 1
    note = round(note_sum / note_count, 2) if note_count else 0
    decisif = round((buts + passes) / selections, 2) if selections else 0
    return {
        "Buts": buts,
        "Passes décisives": passes,
        "Buts + Passes": buts + passes,
        "Décisif par match": decisif,
        "Cartons jaunes": cj,
        "Cartons rouges": cr,
        "Sélections": selections,
        "Titularisations": titularisations,
        "Note générale": note,
        "Homme du match": hdm
    }

def compute_clean_sheets():
    matchs = st.session_state.get("matchs", {})
    clean_sheets = {}
    for match in matchs.values():
        if not match.get("noted"):
            continue
        if match.get("score_adv", 1) > 0:
            continue
        for joueur in match.get("details", {}).get("G", []):
            if joueur and isinstance(joueur, dict) and joueur.get("Nom"):
                name = joueur["Nom"]
                clean_sheets[name] = clean_sheets.get(name, 0) + 1
    return clean_sheets

def get_classement(championnat_scores, adversaires):
    stats = {adv: {"Pts": 0, "V": 0, "N": 0, "D": 0, "BP": 0, "BC": 0} for adv in adversaires + ["AFC"]}
    for journee, matchs in championnat_scores.items():
        for m in matchs:
            dom, ext = m["domicile"], m["exterieur"]
            sd, se = m["score_dom"], m["score_ext"]
            # Mise à jour des scores
            stats[dom]["BP"] += sd
            stats[dom]["BC"] += se
            stats[ext]["BP"] += se
            stats[ext]["BC"] += sd
            # Attribution des points
            if sd > se:
                stats[dom]["V"] += 1
                stats[ext]["D"] += 1
                stats[dom]["Pts"] += 3
            elif se > sd:
                stats[ext]["V"] += 1
                stats[dom]["D"] += 1
                stats[ext]["Pts"] += 3
            else:
                stats[dom]["N"] += 1
                stats[ext]["N"] += 1
                stats[dom]["Pts"] += 1
                stats[ext]["Pts"] += 1
    for v in stats.values():
        v["Diff"] = v["BP"] - v["BC"]
    classement = pd.DataFrame([
        {"Équipe": k, **v} for k, v in stats.items()
    ]).sort_values(["Pts", "Diff", "BP"], ascending=[False, False, False])
    return classement

def style_classement(df):
    styles = []
    for i in range(len(df)):
        if i == 0:
            styles.append(['background-color: #d4edda'] * len(df.columns))  # 🟩 Premier
        elif i >= len(df) - 2:
            styles.append(['background-color: #f8d7da'] * len(df.columns))  # 🟥 Relégables
        else:
            styles.append([''] * len(df.columns))  # ⚪ Milieu de tableau
    return pd.DataFrame(styles, columns=df.columns)

# --- 🧩 Initialiser un terrain vide selon la formation ---
def terrain_init(formation):
    if formation not in FORMATION:
        formation = DEFAULT_FORMATION
    return {poste: [None for _ in range(FORMATION[formation][poste])] for poste in POSTES_ORDER}

# --- 🎮 Interface pour sélectionner les titulaires dynamiquement ---
def terrain_interactif(formation, terrain_key, key_suffix=None, joueurs_disponibles=None):
    if formation not in FORMATION:
        formation = DEFAULT_FORMATION
    
    players_df = st.session_state.players
    if joueurs_disponibles is not None:
        players_df = players_df[players_df["Nom"].isin(joueurs_disponibles)]
    if players_df.empty:
        st.info("Aucun joueur dans la base. Merci d'importer ou d'ajouter des joueurs.")
        return {poste: [] for poste in POSTES_ORDER}

    if terrain_key not in st.session_state:
        st.session_state[terrain_key] = terrain_init(formation)
    terrain = st.session_state[terrain_key]

    stats_data = []
    for _, row in players_df.iterrows():
        s = compute_player_stats(row["Nom"])
        stats_data.append({**row, **s})
    stats_df = pd.DataFrame(stats_data)

    stats_df["Titularisations"] = pd.to_numeric(stats_df.get("Titularisations", 0), errors="coerce").fillna(0)

    key_prefix = f"{terrain_key}_{key_suffix}" if key_suffix else terrain_key
    
    for poste in POSTES_ORDER:
        noms_postes = POSTES_NOMS.get(formation, {}).get(poste, [])
        if not noms_postes:
            noms_postes = [f"{POSTES_LONG[poste]} {i+1}" for i in range(FORMATION[formation][poste])]

        stats_df["is_poste"] = stats_df["Poste"] == poste
        joueurs_tries = stats_df.sort_values(
            by=["is_poste", "Titularisations", "Nom"],
            ascending=[False, False, True]
        )["Nom"].tolist()

        with st.expander(f"{POSTES_LONG[poste] + ('x' if POSTES_LONG[poste] == 'Milieu' else 's')}"):
            for i in range(FORMATION[formation][poste]):
                all_selected = [j["Nom"] for p in POSTES_ORDER for j in terrain.get(p, []) if isinstance(j, dict) and j]
                current = terrain[poste][i]
                current_nom = current["Nom"] if current and isinstance(current, dict) else ""
                label = noms_postes[i] if i < len(noms_postes) else f"{POSTES_LONG[poste]} {i+1}"
                options = [""] + [n for n in joueurs_tries if n == current_nom or n not in all_selected]

                key_select = f"selectbox_{key_prefix}_{poste}_{i}_{current_nom}"
                choix = st.selectbox(label, options, index=options.index(current_nom) if current_nom in options else 0, key=key_select)

                if choix:
                    joueur_info = players_df[players_df["Nom"] == choix].iloc[0].to_dict()
                    num = st.text_input(
                        f"Numéro de {choix}",
                        value=current.get("Numero", "") if current else "",
                        key=f"num_{key_prefix}_{poste}_{i}_{choix}"
                    )
                    joueur_info["Numero"] = num
                    terrain[poste][i] = joueur_info
                else:
                    terrain[poste][i] = None

    st.session_state[terrain_key] = terrain
    return terrain

# --- Gestion des remplaçants dynamique et variable ---
def remplacants_interactif(key, titulaires, key_suffix=None, joueurs_disponibles=None, max_remplacants=MAX_REMPLACANTS):
    players_df = st.session_state.players
    if joueurs_disponibles is not None:
        players_df = players_df[players_df["Nom"].isin(joueurs_disponibles)]
    stats_data = []
    for _, row in players_df.iterrows():
        s = compute_player_stats(row["Nom"])
        stats_data.append({**row, **s})
    stats_df = pd.DataFrame(stats_data)

    stats_df["Titularisations"] = pd.to_numeric(stats_df.get("Titularisations", 0), errors="coerce").fillna(0)
    dispo_base = stats_df.sort_values("Titularisations", ascending=False)["Nom"].tolist()
    dispo = [n for n in dispo_base if n not in titulaires]

    if f"remp_{key}" not in st.session_state or not isinstance(st.session_state[f"remp_{key}"], list):
        st.session_state[f"remp_{key}"] = [{"Nom": None, "Numero": ""} for _ in range(max_remplacants)]
    remps = st.session_state[f"remp_{key}"]

    key_prefix = f"{key}_{key_suffix}" if key_suffix else key

    with st.expander("Remplaçants"):
        for i in range(len(remps)):
            current_nom = remps[i]["Nom"]
            options = dispo + ([current_nom] if current_nom and current_nom not in dispo else [])

            key_select = f"remp_choice_{key_prefix}_{i}_{current_nom}"

            choix = st.selectbox(
                f"Remplaçant {i+1}",
                [""] + options,
                index=(options.index(current_nom) + 1) if current_nom in options else 0,
                key=key_select
            )
            
            if choix:
                num = st.text_input(
                    f"Numéro de {choix}",
                    value=remps[i].get("Numero", ""),
                    key=f"num_remp_{key_prefix}_{i}_{choix}"
                )
                remps[i] = {"Nom": choix, "Numero": num}
            else:
                remps[i] = {"Nom": None, "Numero": ""}

            dispo = [n for n in dispo if n != choix]

        if st.button("➕ Ajouter un remplaçant", key=f"add_remp_{key_prefix}"):
            remps.append({"Nom": None, "Numero": ""})

    st.session_state[f"remp_{key}"] = remps
    return [r for r in remps if r["Nom"]]

# --- 🚀 Initialisation Streamlit globale ---
st.set_page_config(
    page_title="AFC Manager",
    page_icon="⚽",
    layout="wide"
)

# --- 🎨 En-tête visuel ---
st.title("⚽ AFC Manager – Gestion complète de l'équipe")
#st.caption("🧪 Application Streamlit personnalisée pour suivre les performances, les compositions et les résultats du club AFC.")

# --- 🧭 Bouton de rechargement des données (dans la sidebar) ---
#with st.sidebar:
    #st.header("🔧 Options")
    #if st.button("🔄 Recharger les données depuis GitHub"):
        #manager.load()
        #st.success("✅ Données rechargées")
        #st.rerun()

# --- 🧭 Onglets principaux de navigation ---
tab_acc, tab1, tab2, tab3, tab4 = st.tabs([
    "🏠", 
    "📅 Matchs", 
    "📈 Championnat", 
    "👥 Gestion Équipe", 
    "🧠 Tactiques"
])

# --- 🏟️ Onglet Accueil (Tableau de bord) ---
with tab_acc:
    today = datetime.today().date()
    matchs = st.session_state.get("matchs", {})
    classement = get_classement(
        st.session_state.get("championnat_scores", {}),
        st.session_state.get("adversaires", [])
    )
    col1, col2 = st.columns(2)
    # 📊 Classement championnat
    try:
        rang_afc = classement.reset_index(drop=True).query("Équipe == 'AFC'").index[0] + 1
        col1.markdown(
            f"<span style='font-size:22px;'>📊 Championnat :</span> <span style='font-size:36px; font-weight:bold;'>{rang_afc}ᵉ</span>",
            unsafe_allow_html=True
        )
    except IndexError:
        st.warning("AFC ne figure pas encore dans le classement.")

     # 🏆 Parcours en coupe
    match_coupe_a_venir = None
    dernier_tour_coupe = None
    for match in matchs.values():
        if match.get("type", "").lower() == "coupe":
            try:
                date_match = datetime.strptime(match["date"], "%Y-%m-%d").date()
                if date_match >= today:
                    if not match_coupe_a_venir or date_match < datetime.strptime(match_coupe_a_venir["date"], "%Y-%m-%d").date():
                        match_coupe_a_venir = match
                elif not dernier_tour_coupe or date_match > datetime.strptime(dernier_tour_coupe["date"], "%Y-%m-%d").date():
                    dernier_tour_coupe = match
            except:
                continue

    if match_coupe_a_venir:
        col2.markdown(
            f"<span style='font-size:22px;'>🏆 Coupe :</span> <span style='font-size:36px; font-weight:bold;'>{match_coupe_a_venir.get('journee', 'Tour à venir')}</span>",
            unsafe_allow_html=True
        )
    elif dernier_tour_coupe:
        col2.markdown(
            f"<span style='font-size:22px;'>🏆 Coupe :</span> <span style='font-size:36px; font-weight:bold;'>{dernier_tour_coupe.get('journee', 'Tour inconnu')}</span>",
            unsafe_allow_html=True
        )
    else:
        st.info("Coupe pas encore entamée.")
    
    # 📈 Forme récent
    derniers_resultats = []
    for match in sorted(matchs.values(), key=lambda m: m.get("date", ""), reverse=True):
        try:
            date_match = datetime.strptime(match["date"], "%Y-%m-%d").date()
            if date_match < today and match.get("termine") and match.get("noted"):
                score_afc = match.get("score_afc")
                score_adv = match.get("score_adv")
                symbol = "✅" if score_afc > score_adv else "⚖️" if score_afc == score_adv else "❌"
                derniers_resultats.append(symbol)
            if len(derniers_resultats) == 5:
                break
        except:
            continue

    if derniers_resultats:
       col1.markdown(
        f"<span style='font-size:22px;'>📈 Forme récente :</span> <span style='font-size:36px; font-weight:bold;'>{' '.join(derniers_resultats)}</span>",
            unsafe_allow_html=True
       )
    else:
        st.info("Aucun match joué cette saison.")

    # 📅 Prochain match
    prochain_match = None
    date_min = None
    for match in matchs.values():
        try:
            date_match = datetime.strptime(match["date"], "%Y-%m-%d").date()
            if date_match >= today and (not date_min or date_match < date_min):
                prochain_match = match
                date_min = date_match
        except:
            continue

    if prochain_match:
        col2.markdown(
        f"<span style='font-size:22px;'>📅 Prochain match</span> <span style='font-size:36px; font-weight:bold;'>{prochain_match.get('type','')} - {prochain_match.get('journee', '')} - {prochain_match['adversaire']}</span>",
            unsafe_allow_html=True
       )
    else:
        st.info("Aucun match à venir.")


# --- 🧠 Onglet : Gestion Équipe ---
with tab3:
    subtab1, subtab2 = st.tabs(["Stats équipe", "Base joueurs"])

    # -- 🟡 Sous-onglet : Statistiques équipe --
    with subtab1:
        stats_data = []
        for _, row in st.session_state.players.iterrows():
            s = compute_player_stats(row["Nom"])
            stats_data.append({**row, **s})
        df = pd.DataFrame(stats_data)
        clean_sheets = compute_clean_sheets()

        if not df.empty:
            df["Clean sheets"] = df.apply(lambda r: clean_sheets.get(r["Nom"], 0) if r["Poste"] == "G" else None, axis=1)
            df["Bouchers"] = df["Cartons rouges"].fillna(0) + df["Cartons jaunes"].fillna(0)

            # 🎯 Création des tops 5
            top_rating     = df[df["Note générale"] > 0].sort_values("Note générale", ascending=False).head(5)
            top_buts       = df[df["Buts"] > 0].sort_values("Buts", ascending=False).head(5)
            top_passes     = df[df["Passes décisives"] > 0].sort_values("Passes décisives", ascending=False).head(5)
            top_decisive   = df[df["Buts + Passes"] > 0].sort_values("Buts + Passes", ascending=False).head(5)
            top_clean      = df[df["Poste"] == "G"].sort_values("Clean sheets", ascending=False).head(5)
            top_ratio      = df[df["Décisif par match"] > 0].sort_values("Décisif par match", ascending=False).head(5)
            top_used       = df[df["Titularisations"] > 0].sort_values("Titularisations", ascending=False).head(5)
            top_bouchers   = df[(df["Cartons rouges"] > 0) | (df["Cartons jaunes"] > 0)].sort_values(["Cartons rouges", "Cartons jaunes"], ascending=False).head(5)
            top_hdm = df[df["Homme du match"] > 0].sort_values("Homme du match", ascending=False).head(5)

            col1, col2 = st.columns(2)
            with col1:
                
                st.subheader("⚽ Top 5 Buteurs")
                st.dataframe(top_buts[["Nom", "Buts"]], use_container_width=True, hide_index=True)
                st.subheader("🎯 Top 5 Passeurs")
                st.dataframe(top_passes[["Nom", "Passes décisives"]], use_container_width=True, hide_index=True)
                st.subheader("🔥 Top 5 Décisifs")
                st.dataframe(top_decisive[["Nom", "Buts + Passes"]], use_container_width=True, hide_index=True)
                st.subheader("⚡ Ratio par match")
                st.dataframe(top_ratio[["Nom", "Décisif par match"]], use_container_width=True, hide_index=True)
                st.subheader("🧤 Clean Sheets")
                st.dataframe(top_clean[["Nom", "Clean sheets"]], use_container_width=True, hide_index=True)
            with col2:
                st.subheader("⭐ Top 5 Notes")
                st.dataframe(top_rating[["Nom", "Note générale"]], use_container_width=True, hide_index=True)
                st.subheader("🏆 Top 5 Homme du match")
                st.dataframe(top_hdm[["Nom", "Homme du match"]], use_container_width=True, hide_index=True)
                st.subheader("🔁 Plus utilisés")
                st.dataframe(top_used[["Nom", "Titularisations"]], use_container_width=True, hide_index=True)
                st.subheader("🟥🟨 Bouchers")
                st.dataframe(top_bouchers[["Nom", "Cartons rouges", "Cartons jaunes"]], use_container_width=True, hide_index=True)

            # 🏆 Statistiques globales de l'équipe
            st.markdown("---")
            col3, col4, col5 = st.columns(3)
            col3.metric("🧮 Buts marqués", int(df["Buts"].sum()))
            col4.metric("🔓 Buts encaissés", sum(m.get("score_adv", 0) for m in st.session_state.matchs.values() if m.get("noted")))
            col5.metric("👥 Nombre de buteurs", df[df["Buts"] > 0]["Nom"].nunique())
        else:
            st.info("Aucun joueur dans la base pour générer des stats.")

    # -- 🟠 Sous-onglet : Base de données joueurs --
    with subtab2:
        st.markdown("Ajoutez, éditez ou retirez des joueurs ci-dessous. Les colonnes statistiques sont calculées automatiquement.")

        stats_data = []
        for _, row in st.session_state.players.iterrows():
            s = compute_player_stats(row["Nom"])
            stats_data.append({**row, **s})
        df_stats = pd.DataFrame(stats_data, columns=[
            "Nom", "Poste", "Infos", "Buts", "Passes décisives", "Buts + Passes",
            "Décisif par match", "Cartons jaunes", "Cartons rouges", "Sélections",
            "Titularisations", "Note générale", "Homme du match"
        ])

        edited_df = st.data_editor(
            df_stats,
            num_rows="dynamic",
            use_container_width=True,
            hide_index=True,
            column_config={
                "Nom": st.column_config.TextColumn(required=True),
                "Poste": st.column_config.SelectboxColumn(options=POSTES_ORDER, required=True, default="G"),
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
                "Homme du match": st.column_config.NumberColumn(disabled=True),
            },
            key="data_edit_joueurs"
        )

        if st.button("💾", key="btn_save_joueurs"):
            edited_df = edited_df.fillna("")
            edited_df = edited_df[edited_df["Nom"].str.strip() != ""]
            st.session_state.players = edited_df[PLAYER_COLS]
            manager.save()
            st.success("Base de joueurs mise à jour ✅")
            st.rerun()
        st.caption("🗑️ Pour supprimer un joueur, videz son nom et cliquez sur Sauvegarder.")

# --- 🧪 Onglet Tactiques ---
with tab4:

    subtab1, subtab2, subtab3 = st.tabs([
        "Créer une composition", 
        "Mes compositions", 
        "Profondeur d'effectif"
    ])

    # --- 📐 Créer une composition ---
    with subtab1:
        edit_key = "edit_compo"
        edit_compo = st.session_state.get(edit_key, None)

        # Mode édition si compo à éditer
        if edit_compo:
            nom_compo, loaded = edit_compo
            st.info(f"✏️ Édition de la composition : **{nom_compo}**")
            st.session_state["formation_create_compo"] = loaded["formation"]
            st.session_state["terrain_create_compo_tactic"] = loaded["details"]
            del st.session_state[edit_key]

        nom_compo = st.text_input("📝 Nom de la composition", key="nom_compo_create", value=nom_compo if edit_compo else "")
        formation = st.selectbox("🏟 Formation", list(FORMATION.keys()), index=list(FORMATION.keys()).index(st.session_state.get("formation_create_compo", DEFAULT_FORMATION)), key="formation_create_compo")

        col_left, col_right = st.columns([1, 2])
        with col_left:
            terrain = terrain_interactif(formation, "terrain_create_compo_tactic")
            titulaires = [j["Nom"] for p in POSTES_ORDER for j in terrain.get(p, []) if j]
            remplacants = remplacants_interactif("create_compo", titulaires)

        with col_right:
            fig = draw_football_pitch_vertical()
            fig = plot_lineup_on_pitch_vertical(fig, terrain, formation, remplacants)
            st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key="fig_create_compo")

        # Sauvegarde
        if st.button("💾"):
            if not nom_compo.strip():
                st.error("🚫 Veuillez indiquer un nom pour la composition.")
                st.stop()
            try:
                lineup = {
                    "formation": formation,
                    "details": terrain,
                    "remplacants": remplacants
                }
                st.session_state.lineups[nom_compo] = lineup
                manager.save()
                st.success("✅ Composition enregistrée !")
                st.rerun()
            except Exception as e:
                st.error(f"❌ Erreur : {e}")

    # --- 🗂️ Compositions existantes ---
    with subtab2:
        if not st.session_state.lineups:
            st.info("📭 Aucune composition enregistrée.")
        else:
            for nom, compo in st.session_state.lineups.items():
                with st.expander(f"{nom} – {compo['formation']}"):
                    fig = draw_football_pitch_vertical()
                    fig = plot_lineup_on_pitch_vertical(fig, compo["details"], compo["formation"], compo.get("remplacants", []))
                    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"fig_compo_{nom}")

                    col1, col2 = st.columns(2)
                    if col1.button(f"✏️", key=f"edit_{nom}"):
                        st.session_state["edit_compo"] = (nom, compo)
                        st.rerun()
                    if col2.button(f"🗑️", key=f"delete_{nom}"):
                        del st.session_state.lineups[nom]
                        manager.save()
                        st.success("✅ Composition supprimée")
                        st.rerun()

    # --- 🔎 Profondeur d'effectif par poste ---
    with subtab3 : 
        formation_selected = st.selectbox("🎯 Formation", list(FORMATION.keys()), key="formation_profondeur")
        
        # 🔧 Initialisation
        if "profondeur_effectif" not in st.session_state:
            st.session_state.profondeur_effectif = {}
        if formation_selected not in st.session_state.profondeur_effectif:
            st.session_state.profondeur_effectif[formation_selected] = {}
        
        profondeur = st.session_state.profondeur_effectif[formation_selected]
        joueurs = st.session_state.players["Nom"].dropna().tolist()
        postes_formation = POSTES_NOMS.get(formation_selected, {})
        positions = positions_for_formation_vertical(formation_selected)
        
        # 🧭 Deux colonnes : Sélection / Terrain
        col_left, col_right = st.columns([3, 7])
        
        with col_left:
            st.markdown("### Sélection par poste")
            for poste in POSTES_ORDER:
                if poste not in postes_formation:
                    continue
                if poste not in profondeur:
                    profondeur[poste] = {}
                for idx_label, label in enumerate(postes_formation[poste]):
                    idx_key = str(idx_label)
                    if idx_key not in profondeur[poste]:
                        profondeur[poste][idx_key] = [""]
                    with st.expander(f"**{label}**"):
                        new_noms = []
                        for i, nom in enumerate(profondeur[poste][idx_key]):
                            select_key = f"{formation_selected}_{poste}_{idx_key}_choix_{i}_depth"
                            choix = st.selectbox(
                                f"Option {i+1}", [""] + joueurs,
                                index=([""] + joueurs).index(nom) if nom in joueurs else 0,
                                key=select_key
                            )
                            if choix.strip():
                                new_noms.append(choix)
                        if new_noms and new_noms[-1] != "":
                            new_noms.append("")
                        profondeur[poste][idx_key] = new_noms
        
            if st.button("💾", key=f"save_profondeur_{formation_selected}"):
                st.session_state.profondeur_effectif[formation_selected] = profondeur
                manager.save()
                st.success("✅ Profondeur sauvegardée")
        
        with col_right:
            st.markdown("### Visualisation du terrain")
            fig = draw_football_pitch_vertical()
            for poste in POSTES_ORDER:
                for idx, label in enumerate(postes_formation.get(poste, [])):
                    noms = profondeur.get(poste, {}).get(str(idx), [])
                    noms = [n for n in noms if n]
                    if noms:
                        x, y = positions[poste][idx % len(positions[poste])]
                        texte = "<br>".join([f"{i+1}. {nom}" for i, nom in enumerate(noms)])
                        fig.add_annotation(
                            x=x, y=y,
                            text=texte,
                            showarrow=False,
                            font=dict(size=13, color="white"),
                            bgcolor="#0d47a1",
                            bordercolor="white",
                            borderwidth=1,
                            borderpad=4,
                            align="center"
                        )
            st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key="fig_profondeur")

# --- 📅 Onglet Gestion des matchs ---
with tab1:
    subtab1, subtab2 = st.tabs(["Mes matchs", "Créer un match"])

    # --- ⚙️ Créer un match ---
    with subtab2:
        with subtab2:
            edit_match = st.session_state.get("edit_match", None)
            if edit_match:
                mid_edit, match_data = edit_match
                st.info(f"✏️ Édition du match : **{match_data['nom_match']}**")
                type_match = st.selectbox("Compétition", ["Championnat", "Coupe", "Amical"], index=["Championnat", "Coupe", "Amical"].index(match_data["type"]))
                if type_match=="Championnat":
                    journee = st.text_input("Journée", value=match_data["journee"])
                else:
                    if type_match=="Coupe":
                        journee=st.selectbox("Tour", ["Poules", "Huitièmes", "Quarts", "Demies", "Finale"], index=["Poules", "Huitièmes", "Quarts", "Demies", "Finale"].index(match_data["journee"]))
                    else:
                        journee = st.text_input("Numéro", value=match_data["journee"])
                adversaires_list = st.session_state.get("adversaires", [])
                options_adversaires = adversaires_list + ["Autre..."]
                default_adv = match_data["adversaire"] if match_data["adversaire"] in adversaires_list else "Autre..."
                adversaire_select = st.selectbox("Adversaire", options_adversaires, index=options_adversaires.index(default_adv))    
                if adversaire_select == "Autre...":
                    adversaire = st.text_input("🆕 Nom de l'adversaire", value=match_data["adversaire"])
                else:
                    adversaire = adversaire_select
                date = st.date_input("Date du match", value=match_data["date"])
                heure = st.time_input("Heure du match", value=match_data["heure"])
                domicile = st.selectbox("Réception", ["Domicile", "Extérieur"],index=["Domicile", "Extérieur"].index(match_data["domicile"]))
                if domicile == "Domicile":
                    lieu_default = "Club de Football Barradels, 2 Rue des Cyclamens, 31700 Blagnac"
                else:
                    lieu_default = match_data.get("lieu", "")
                lieu = st.text_input("Lieu du match", value=lieu_default)
            else:
                type_match = st.selectbox("Compétition", ["Championnat", "Coupe", "Amical"])
                if type_match=="Championnat":
                    journee = st.text_input("Journée", value="J")
                else:
                    if type_match=="Coupe":
                        journee=st.selectbox("Tour", ["Poules", "Huitièmes", "Quarts", "Demies", "Finale"])
                    else:
                        journee = st.text_input("Numéro", value="#")
                adversaires_list = st.session_state.get("adversaires", [])
                adversaire_select = st.selectbox("Adversaire", adversaires_list + ["Autre..."])
                if adversaire_select == "Autre...":
                    adversaire = st.text_input("🆕 Nom de l'adversaire")
                else:
                    adversaire = adversaire_select
        
                date = st.date_input("Date du match", value=datetime.today())
                heure = st.time_input("Heure du match", value=datetime.strptime("21:00", "%H:%M").time())
                domicile = st.selectbox("Réception", ["Domicile", "Extérieur"])
                if domicile == "Domicile":
                    lieu_default = "Club de Football Barradels, 2 Rue des Cyclamens, 31700 Blagnac"
                else:
                    lieu_default = ""
                lieu = st.text_input("Lieu du match", value=lieu_default)
            if type_match=="Championnat":
                nom_match = f"📈 {date} : {type_match} - {journee} - {'AFC vs' if domicile == 'Domicile' else ''} {adversaire}{' vs AFC' if domicile == 'Extérieur' else ''}"
            else:
                if type_match=="Coupe":
                    nom_match = f"🏆 {date} : {type_match} - {journee} - {'AFC vs' if domicile == 'Domicile' else ''} {adversaire}{' vs AFC' if domicile == 'Extérieur' else ''}"
                else:
                    nom_match = f"🤝 {date} : {type_match} - {journee} - {'AFC vs' if domicile == 'Domicile' else ''} {adversaire}{' vs AFC' if domicile == 'Extérieur' else ''}"
        
            if st.button("💾", key="save_match"):
                match_data = {
                    "type": type_match,
                    "adversaire": adversaire,
                    "date": str(date),
                    "heure": heure.strftime("%H:%M"),
                    "domicile": domicile,
                    "journee": journee,
                    "nom_match": nom_match,
                    "lieu": lieu,
                    "formation": "",
                    "details": [],
                    "remplacants": [],
                    "events": {},
                    "score": "",
                    "score_afc": 0,
                    "score_adv": 0,
                    "noted": False,
                    "termine": False,
                    "homme_du_match": ""
                }
                if edit_match:
                    st.session_state.matchs[mid_edit] = match_data
                    del st.session_state["edit_match"]
                    st.success("✅ Match mis à jour")
                else:
                    mid = str(uuid.uuid4())
                    st.session_state.matchs[mid] = match_data
                    st.success("✅ Match enregistré")
                manager.save()
                st.rerun()

    # --- 📋 Mes matchs enregistrés ---
    with subtab1:
        if not st.session_state.matchs:
            st.info("📭 Aucun match enregistré.")
        else:
            matchs_tries = sorted(st.session_state.matchs.items(), key=lambda x: x[1].get("date", ""), reverse=False)
            for mid, match in matchs_tries:
                with st.expander(match.get("nom_match", "Match sans nom")):
    
                    # --- ✅ Checkbox “Match terminé” ---
                    match_ended = st.checkbox("Match terminé", value=match.get("termine", False), key=f"ended_{mid}")
    
                    if match_ended != match.get("termine", False):
                        match["termine"] = match_ended
                        st.session_state.matchs[mid] = match
                        manager.save()
                        st.rerun()
    
                    if not match.get("termine"):
                        # --- 👥 Sélection des joueurs disponibles ---
                        players_df = st.session_state.players.copy()
                        players_df = players_df.sort_values(["Poste", "Nom"])
                        joueurs_tries = players_df["Nom"].tolist()
                        if "joueurs_disponibles" not in match:
                            match["joueurs_disponibles"] = players_df.copy()
                        with st.expander("👥 Sélection des joueurs disponibles pour ce match"):
                            selected_dispo = st.multiselect(
                                "Joueurs disponibles",
                                joueurs_tries,
                                default=match.get("joueurs_disponibles", []),
                                key=f"joueurs_dispo_{mid}"
                            )
                            match["joueurs_disponibles"] = selected_dispo
                            st.session_state.matchs[mid] = match
                            st.markdown(f"Joueurs disponibles : {len(selected_dispo)}/{len(joueurs_tries)}")
    
                            if st.button("💾 Sauvegarder la liste des joueurs disponibles", key=f"save_dispo_{mid}"):
                                match["joueurs_disponibles"] = selected_dispo
                                st.session_state.matchs[mid] = match
                                manager.save()
                                st.success("Liste des joueurs disponibles sauvegardée !")
        
                        # --- 🏟️ Créer composition pour ce match ---
                        with st.expander("### 🏟️ Composition du match"):
                            formation = match.get("formation", DEFAULT_FORMATION)
                            terrain_key = f"terrain_match_{mid}"
                            remp_key = f"remp_match_{mid}"
                            if terrain_key not in st.session_state and match.get("details"):
                                st.session_state[terrain_key] = match["details"]
                            if remp_key not in st.session_state and match.get("remplacants"):
                                st.session_state[remp_key] = match["remplacants"]
                            joueurs_dispo = match.get("joueurs_disponibles", [])
                            
                            col_left, col_right = st.columns([3, 7])
                            joueurs_dispo = match.get("joueurs_disponibles", [])
                            # Nombre de remplaçants variable pour amical
                            if match["type"] == "Amical":
                                max_remplacants = st.number_input(
                                    "Nombre de remplaçants",
                                    min_value=5,
                                    max_value=20,
                                    value=len(match.get("remplacants", [])) if match.get("remplacants") else 5,
                                    key=f"max_remp_{mid}"
                                )
                            else:
                                max_remplacants = MAX_REMPLACANTS
                            with col_left:
                                use_compo = st.checkbox("🔁 Utiliser une compo enregistrée ?", key=f"use_compo_{mid}")
                                if use_compo and st.session_state.lineups:
                                    compo_choice = st.selectbox("📂 Choisir une compo", list(st.session_state.lineups.keys()), key=f"compo_choice_{mid}")
                                    compo_data = st.session_state.lineups[compo_choice]
                                    formation = compo_data["formation"]
                                    terrain = compo_data["details"]
                                    remplacants = compo_data.get("remplacants", [])
                                else:
                                    formation = st.selectbox(
                                        "📌 Formation",
                                        list(FORMATION.keys()),
                                        key=f"form_{mid}",
                                        index=list(FORMATION.keys()).index(match.get("formation", DEFAULT_FORMATION)) if match.get("formation") else 0
                                    )
                                    terrain = terrain_interactif(
                                        formation,
                                        terrain_key,
                                        key_suffix=mid,
                                        joueurs_disponibles=joueurs_dispo
                                    )
                                    titulaires = [j["Nom"] for p in POSTES_ORDER for j in terrain.get(p, []) if j]
                                    remplacants = remplacants_interactif(
                                        f"match_{mid}",
                                        titulaires,
                                        key_suffix=mid,
                                        joueurs_disponibles=joueurs_dispo,
                                        max_remplacants=len(match.get("remplacants", [])) if match.get("remplacants") else MAX_REMPLACANTS
                                    )
                                # Sélection du capitaine à la fin (parmi les titulaires)
                                titulaires_noms = [j["Nom"] for p in POSTES_ORDER for j in terrain.get(p, []) if j]
                                capitaine = st.selectbox(
                                    "🧑‍✈️ Capitaine",
                                    [""] + titulaires_noms,
                                    index=(titulaires_noms.index(match.get("capitaine", "")) + 1) if match.get("capitaine", "") in titulaires_noms else 0,
                                    key=f"cap_match_{mid}"
                                )
                                
                            with col_right:
                                fig = draw_football_pitch_vertical()
                                fig = plot_lineup_on_pitch_vertical(fig, terrain, formation, remplacants)
                                st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"fig_match_{mid}")
    
                                # Visualisation des joueurs disponibles non retenus
                                joueurs_dispo_set = set(joueurs_dispo)
                                retenus = set(titulaires_noms + [r.get("Nom") for r in remplacants if r.get("Nom")])
                                non_retenus = joueurs_dispo_set - retenus
                                if non_retenus:
                                    st.markdown("**Non retenus :** " + ", ".join(non_retenus))
    
                            if st.button("💾", key=f"btn_compo_{mid}"):
                                match["formation"] = formation
                                match["details"] = terrain
                                match["remplacants"] = remplacants
                                match["capitaine"] = capitaine
                                st.session_state.matchs[mid] = match
                                manager.save()
                                st.success("✅ Composition enregistrée")
                                st.rerun()
    
                        # --- 👥 Convocation des joueurs ---
                        
                        st.markdown("""
                            <style>
                            .convoc-container {
                                background: linear-gradient(135deg, #0a2342, #1d3557);
                                color: white;
                                padding: 20px;
                                border-radius: 10px;
                                text-align: center;
                            }
                            .convoc-title {
                                font-size: 28px;
                                font-weight: bold;
                                color: #ffcc00;
                                margin-bottom: 10px;
                            }
                            .convoc-sub {
                                font-size: 20px;
                                margin-bottom: 20px;
                            }
                            .poste {
                                font-size: 22px;
                                font-weight: bold;
                                margin-top: 15px;
                                margin-bottom: 5px;
                                color: #ffcc00;
                                border-bottom: 2px solid #ffcc00;
                                display: inline-block;
                                padding: 2px 10px;
                            }
                            .joueurs {
                                display: flex;
                                justify-content: center;
                                flex-wrap: wrap;
                                gap: 12px;
                                margin-bottom: 15px;
                            }
                            .joueur {
                                background: #2c3e50;
                                border-radius: 6px;
                                padding: 8px 12px;
                                font-size: 16px;
                                font-weight: 600;
                            }
                            </style>
                        """, unsafe_allow_html=True)
                        if terrain:
                            with st.expander("### 👥 Convocation des joueurs"):
                                noms_titulaires = [j["Nom"] for p in POSTES_ORDER for j in terrain.get(p, []) if j]
                                noms_remplaçants = [r["Nom"] for r in remplacants if r.get("Nom")]
                                noms_convoques = list(dict.fromkeys(noms_titulaires + noms_remplaçants))
    
                                joueurs_df = st.session_state.players.set_index("Nom")
                                convoques_par_poste = {poste: [] for poste in POSTES_ORDER}
                                for nom in noms_convoques:
                                    poste = joueurs_df.at[nom, "Poste"] if nom in joueurs_df.index else ""
                                    if poste in convoques_par_poste:
                                        convoques_par_poste[poste].append(nom)
                                    else:
                                        convoques_par_poste.setdefault("?", []).append(nom)
    
                                heure_match = match.get("heure", "21:00")
                                try:
                                    rdv = (datetime.strptime(heure_match, "%H:%M") - timedelta(hours=1)).strftime("%H:%M")
                                except:
                                    rdv = "?"
                                 # --- 📋 Affiche convocation ---
                                st.markdown(f"""
                                    <div class="convoc-container">
                                        <div class="convoc-title">📋 Convocation - {match['type']} {match['journee']}</div>
                                        <div class="convoc-sub">🆚 {match.get('adversaire')}<br>
                                        📅 {match['date']} à {match['heure']} – RDV {rdv}<br>
                                        {"🏠" if match['domicile']=="Domicile" else "🚗"} {match.get('lieu')}</div>
                                """, unsafe_allow_html=True)
                        
                                # --- Par poste ---
                                POSTES_EMOJIS = {"G": "🧤", "D": "🛡️", "M": "🎯", "A": "⚽"}
                                for poste in POSTES_ORDER:
                                    joueurs = convoques_par_poste.get(poste, [])
                                    if joueurs:
                                        label = POSTES_LONG.get(poste, "Inconnu")
                                        emoji = POSTES_EMOJIS.get(poste, "")
                                        st.markdown(f"<div class='poste'>{emoji} {label + ('x' if label == 'Milieu' else 's')}</div>", unsafe_allow_html=True)
                        
                                        joueurs_html = "".join([f"<div class='joueur'>{nom}</div>" for nom in joueurs])
                                        st.markdown(f"<div class='joueurs'>{joueurs_html}</div>", unsafe_allow_html=True)
                        
                                # Fermeture du container
                                st.markdown("</div>", unsafe_allow_html=True)
    
                    # --- 📝 Saisie des statistiques du match ---
                    elif match_ended and not match.get("noted", False):
                        with st.expander("### 📝 Statistiques du match"):
                            joueurs = [j["Nom"] for p in POSTES_ORDER for j in match.get("details", {}).get(p, []) if j]
                            joueurs += [r["Nom"] for r in match.get("remplacants", []) if r.get("Nom")]
                            joueurs = list(dict.fromkeys(joueurs))
                    
                            col_eqdom, col_scoredom, col_scoreext, col_eqext = st.columns([3, 2, 2, 3])
                            if match['domicile'] == "domicile":
                                col_eqdom.markdown("AFC")
                                col_eqext.markdown(f"{match['adversaire']}")
                                score_afc = col_scoredom.number_input("⚽", min_value=0, max_value=20, value=0, key=f"score_afc_{mid}")
                                score_adv = col_scoreext.number_input(f"⚽", min_value=0, max_value=20, value=0, key=f"score_adv_{mid}")
                            else:
                                col_eqdom.markdown(f"{match['adversaire']}")
                                col_eqext.markdown("AFC")
                                score_afc = col_scoreext.number_input("⚽", min_value=0, max_value=20, value=0, key=f"score_afc_{mid}")
                                score_adv = col_scoredom.number_input(f"⚽", min_value=0, max_value=20, value=0, key=f"score_adv_{mid}")
                    
                            # Structure événements
                            events = {
                                "buteurs": {},
                                "passeurs": {},
                                "cartons_jaunes": {},
                                "cartons_rouges": {},
                                "notes": {}
                            }
                    
                            # --- Gestion des buts AFC ---
                            st.subheader("⚽ Buts AFC")
                            for i in range(score_afc):
                                col_but1, col_but2 = st.columns([2, 2])
                                buteur = col_but1.selectbox(
                                    f"Buteur du but {i+1}",
                                    [""] + joueurs,
                                    key=f"buteur_{mid}_{i}"
                                )
                                passeur = col_but2.selectbox(
                                    f"Passeur du but {i+1}",
                                    [""] + joueurs,
                                    key=f"passeur_{mid}_{i}"
                                )
                                if buteur:
                                    events["buteurs"][buteur] = events["buteurs"].get(buteur, 0) + 1
                                if passeur:
                                    events["passeurs"][passeur] = events["passeurs"].get(passeur, 0) + 1
                    
                            st.markdown("---")
                    
                            # --- Gestion des cartons ---
                            st.subheader("🟨🟥 Cartons")
                            if "cartons" not in st.session_state:
                                st.session_state["cartons"] = {"jaunes": [], "rouges": []}
                    
                            # Ajouter un carton jaune
                            if st.button("Ajouter un carton jaune", key=f"add_cj_{mid}"):
                                st.session_state["cartons"]["jaunes"].append("")
                            # Ajouter un carton rouge
                            if st.button("Ajouter un carton rouge", key=f"add_cr_{mid}"):
                                st.session_state["cartons"]["rouges"].append("")
                    
                            # Afficher les cartons jaunes
                            for idx, cj in enumerate(st.session_state["cartons"]["jaunes"]):
                                col_cj1, col_cj2 = st.columns([3, 1])
                                joueur_cj = col_cj1.selectbox(
                                    f"🟨 Carton jaune {idx+1}",
                                    [""] + joueurs,
                                    index=([""] + joueurs).index(cj) if cj in joueurs else 0,
                                    key=f"cj_{mid}_{idx}"
                                )
                                st.session_state["cartons"]["jaunes"][idx] = joueur_cj
                                if col_cj2.button("❌", key=f"del_cj_{mid}_{idx}"):
                                    st.session_state["cartons"]["jaunes"].pop(idx)
                                    st.rerun()
                    
                            # Afficher les cartons rouges
                            for idx, cr in enumerate(st.session_state["cartons"]["rouges"]):
                                col_cr1, col_cr2 = st.columns([3, 1])
                                joueur_cr = col_cr1.selectbox(
                                    f"🟥 Carton rouge {idx+1}",
                                    [""] + joueurs,
                                    index=([""] + joueurs).index(cr) if cr in joueurs else 0,
                                    key=f"cr_{mid}_{idx}"
                                )
                                st.session_state["cartons"]["rouges"][idx] = joueur_cr
                                if col_cr2.button("❌", key=f"del_cr_{mid}_{idx}"):
                                    st.session_state["cartons"]["rouges"].pop(idx)
                                    st.rerun()
                    
                            # Mise à jour events avec cartons
                            for nom in st.session_state["cartons"]["jaunes"]:
                                if nom:
                                    events["cartons_jaunes"][nom] = events["cartons_jaunes"].get(nom, 0) + 1
                            for nom in st.session_state["cartons"]["rouges"]:
                                if nom:
                                    events["cartons_rouges"][nom] = events["cartons_rouges"].get(nom, 0) + 1
                    
                            st.markdown("---")
                    
                            # --- Notes joueurs ---
                            noter_joueurs = st.checkbox("Noter les joueurs ?", value=match.get("noter_joueurs", True), key=f"noter_{mid}")
                            match["noter_joueurs"] = noter_joueurs
                    
                            if noter_joueurs:
                                st.subheader("📊 Notes des joueurs")
                                for nom in joueurs:
                                    events["notes"][nom] = st.slider(
                                        f"{nom}",
                                        min_value=0.0, max_value=10.0,
                                        value=5.0, step=0.5,
                                        key=f"note_{mid}_{nom}"
                                    )
                    
                            st.markdown("---")
                            hdm = st.selectbox("🏆 Homme du match", [""] + joueurs, key=f"hdm_{mid}")
    
                            if st.button("💾", key=f"valide_{mid}"):
                                match["events"] = events
                                match["score_afc"] = score_afc
                                match["score_adv"] = score_adv
                                match["score"] = f"{score_afc}-{score_adv}"
                                match["noted"] = True
                                match["termine"] = True
                                match["homme_du_match"] = hdm
                                st.session_state.matchs[mid] = match
                                manager.save()
                                st.success("✅ Statistiques enregistrées")
                                st.rerun()
    
                    # --- 🧾 Résumé si match noté ---
                    elif match.get("noted", False):
                        with st.expander("### 📝 Résumé du match"):
                            st.title(f"{match['nom_match']}")
                            if match.get("domicile") == "Domicile":
                                st.markdown(f"### AFC {match['score_afc']} - {match['score_adv']} {match['adversaire']}")
                            else:
                                st.markdown(f"### {match['adversaire']} {match['score_adv']} - {match['score_afc']} AFC")
    
                            st.markdown("---")
                            hdm = match.get("homme_du_match")
                            if hdm:
                                st.markdown("#### 🏆 Homme du match")
                                st.markdown(f"{hdm}")
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("#### 📊 Événements du match")
                                if match["events"].get("buteurs"):
                                    st.markdown("**⚽ Buteurs**")
                                    for nom, nb in match["events"]["buteurs"].items():
                                        if isinstance(nb, int) and nb > 0:
                                            st.markdown(f"- {nom} : {nb}")
                                if match["events"].get("passeurs"):
                                    st.markdown("**🎯 Passeurs**")
                                    for nom, nb in match["events"]["passeurs"].items():
                                        if isinstance(nb, int) and nb > 0:
                                            st.markdown(f"- {nom} : {nb}")
                            with col2:
                                st.markdown("#### 👮🏼‍♂️ Discipline")
                                if match["events"].get("cartons_jaunes"):
                                    st.markdown("**🟨 Cartons jaunes**")
                                    for nom, nb in match["events"]["cartons_jaunes"].items():
                                        if isinstance(nb, int) and nb > 0:
                                            st.markdown(f"- {nom} : {nb}")
                                if match["events"].get("cartons_rouges"):
                                    st.markdown("**🟥 Cartons rouges**")
                                    for nom, nb in match["events"]["cartons_rouges"].items():
                                        if isinstance(nb, int) and nb > 0:
                                            st.markdown(f"- {nom} : {nb}")
                            st.markdown("---")
    
                            fig = draw_football_pitch_vertical()
    
                            # Prépare les stats pour tous les joueurs (y compris remplaçants)
                            joueurs_titulaires = [j["Nom"] for p in POSTES_ORDER for j in match["details"].get(p, []) if j]
                            joueurs_remplacants = [r["Nom"] for r in match.get("remplacants", []) if isinstance(r, dict) and r.get("Nom")]
                            joueurs_all = list(dict.fromkeys(joueurs_titulaires + joueurs_remplacants))
                            player_stats = {
                                nom: {
                                    "buts": match["events"]["buteurs"].get(nom, 0),
                                    "passes": match["events"]["passeurs"].get(nom, 0),
                                    "cj": match["events"]["cartons_jaunes"].get(nom, 0),
                                    "cr": match["events"]["cartons_rouges"].get(nom, 0),
                                    "note": match["events"]["notes"].get(nom, 0) if match.get("noter_joueurs", True) else None,
                                    "hdm": match.get("homme_du_match") == nom
                                }
                                for nom in joueurs_all
                            }
                            # Affichage du capitaine
                            for poste in POSTES_ORDER:
                                for j in match["details"].get(poste, []):
                                    if j and isinstance(j, dict) and j["Nom"] == match.get("capitaine", ""):
                                        j["Capitaine"] = True
    
                            fig = plot_lineup_on_pitch_vertical(
                                fig,
                                match["details"],
                                match["formation"],
                                match.get("remplacants", []),
                                player_stats=player_stats
                            )
    
                            st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"fig_match_{mid}")
                            if st.button("✏️", key=f"edit_stats_{mid}"):
                                match["noted"] = False
                                st.session_state.matchs[mid] = match
                                manager.save()
                                st.rerun()
    
                    col_gauche, col_droite, col_space = st.columns([0.5, 0.5, 9])
                    if not match.get("termine"):
                        with col_gauche:
                            if st.button("✏️", key=f"btn_edit_{mid}"):
                                st.session_state["edit_match"] = (mid, match)
                                st.rerun()
                        with col_droite:
                            if st.button("🗑️", key=f"delete_match_{mid}"):
                                del st.session_state.matchs[mid]
                                manager.save()
                                st.success("🧹 Match supprimé")
                                st.rerun()
                    else:
                        with col_gauche:
                            if st.button("🗑️", key=f"delete_match_{mid}"):
                                del st.session_state.matchs[mid]
                                manager.save()
                                st.success("🧹 Match supprimé")
                                st.rerun()

# --- 📈 Onglet Suivi Championnat ---
with tab2:
    subtab1, subtab2, subtab3 = st.tabs([
        "Classement", 
        "Saisie des scores", 
        "Gestion des adversaires"
    ])

    # --- 🏆 Classement automatique ---
    with subtab1:
        equipes = ["AFC"] + st.session_state.get("adversaires", [])
        scores = st.session_state.get("championnat_scores", {})
        stats = {team: {"MJ": 0, "Pts": 0, "V": 0, "N": 0, "D": 0, "BP": 0, "BC": 0} for team in equipes}

        for journee, matchs in scores.items():
            for m in matchs:
                dom, ext = m["domicile"], m["exterieur"]
                sd, se = m["score_dom"], m["score_ext"]

                if sd is not None and se is not None:
                    stats[dom]["MJ"] += 1
                    stats[ext]["MJ"] += 1
                    stats[dom]["BP"] += sd
                    stats[dom]["BC"] += se
                    stats[ext]["BP"] += se
                    stats[ext]["BC"] += sd

                    # Résultat
                    if sd > se:
                        stats[dom]["V"] += 1
                        stats[ext]["D"] += 1
                        stats[dom]["Pts"] += 3
                    elif se > sd:
                        stats[ext]["V"] += 1
                        stats[dom]["D"] += 1
                        stats[ext]["Pts"] += 3
                    else:
                        stats[dom]["N"] += 1
                        stats[ext]["N"] += 1
                        stats[dom]["Pts"] += 1
                        stats[ext]["Pts"] += 1

        for v in stats.values():
            v["Diff"] = v["BP"] - v["BC"]

        df = pd.DataFrame([
            {"Équipe": k, **v} for k, v in stats.items()
        ]).sort_values(["Pts", "Diff", "BP"], ascending=[False, False, False])

        st.dataframe(df, hide_index=True, use_container_width=True)

    # --- 📋 Saisie des scores par journée ---
    with subtab2:
        def next_journee_key():
            existing = [int(j[1:]) for j in st.session_state.championnat_scores if j.startswith("J")]
            return f"J{max(existing, default=0)+1:02d}"

        if not st.session_state.championnat_scores:
            st.session_state.championnat_scores["J01"] = []

        journees = sorted(st.session_state.championnat_scores.keys())
        if "selected_journee" not in st.session_state:
            st.session_state.selected_journee = journees[0]
        selected = st.session_state.selected_journee
            
        idx = journees.index(st.session_state.selected_journee)
        col_spacer1, col_prev, col_title, col_next, col_spacer2 = st.columns([3, 1, 2, 1, 3])

        with col_prev:
            if idx > 0 and st.button("←", key="journee_prev"):
                st.session_state.selected_journee = journees[idx - 1]
                st.rerun()
        
        with col_title:
            st.markdown(
                f"<h2 style='text-align: center;'>📅 {selected}</h2>",
                unsafe_allow_html=True
            )
        
        with col_next:
            if idx < len(journees) - 1 and st.button("→", key="journee_next"):
                st.session_state.selected_journee = journees[idx + 1]
                st.rerun()

        selected = st.session_state.selected_journee
        matchs = st.session_state.championnat_scores.get(selected, [])

        # Affichage/édition des matchs
        for i, match in enumerate(matchs):
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 3, 1])
            dom = col1.selectbox(f"Domicile {i+1}", equipes, index=equipes.index(match["domicile"]), key=f"dom_{selected}_{i}")
            score_dom = col2.number_input("⚽", value=match.get("score_dom", 0), min_value=0, max_value=30, key=f"score_dom_{selected}_{i}")
            score_ext = col3.number_input("⚽", value=match.get("score_ext", 0), min_value=0, max_value=30, key=f"score_ext_{selected}_{i}")
            ext = col4.selectbox(f"Extérieur {i+1}", equipes, index=equipes.index(match["exterieur"]), key=f"ext_{selected}_{i}")
            matchs[i] = {
                "domicile": dom,
                "score_dom": score_dom,
                "exterieur": ext,
                "score_ext": score_ext
            }
            
            if col5.button("🗑️", key=f"delete_match_{selected}_{i}"):
                del matchs[i]
                st.session_state.championnat_scores[selected] = matchs
                manager.save()
                st.success("🧹 Match supprimé")
                st.rerun()
                
        uniquekeysave = f"savescores_{selected}"
        if st.button("💾",key=uniquekeysave):
                st.session_state.championnat_scores[selected] = matchs
                manager.save()
                classement = get_classement(
                    st.session_state.championnat_scores,
                    st.session_state.adversaires
                )
                st.session_state["classement_live"] = classement  # (Optionnel, si tu veux le réutiliser ailleurs)
                st.success("✅ Scores mis à jour")
                st.rerun()

        # Ajouter un match
        st.markdown("---")
        with st.expander("➕ Ajouter un match à cette journée"):
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 3, 1])
            dom_new = col1.selectbox("Domicile", equipes)
            ext_new = col4.selectbox("Extérieur", [e for e in equipes if e != dom_new])
            score_dom_new = col2.number_input("⚽ Score domicile", min_value=0, value=0)
            score_ext_new = col3.number_input("⚽ Score extérieur", min_value=0, value=0)
            if col5.button("➕"):
                matchs.append({
                    "domicile": dom_new,
                    "exterieur": ext_new,
                    "score_dom": score_dom_new,
                    "score_ext": score_ext_new
                })
                st.session_state.championnat_scores[selected] = matchs
                manager.save()
                st.success("✅ Match ajouté")
                st.rerun()
        st.markdown("---")
        # Ajouter une nouvelle journée
        if st.button("🗓️ Ajouter une journée"):
            new_key = next_journee_key()
            st.session_state.championnat_scores[new_key] = []
            manager.save()
            st.success(f"📅 {new_key} créée")
            st.rerun()

    # --- 🧑‍🤝‍🧑 Gestion des adversaires ---
    with subtab3:
        adv_df = pd.DataFrame({"Nom": st.session_state.adversaires}, dtype="object")
        edited_df = st.data_editor(adv_df, num_rows="dynamic", hide_index=True)

        if st.button("💾",key=f"save_{adv_df}"):
            st.session_state.adversaires = edited_df["Nom"].dropna().astype(str).tolist()
            manager.save()
            st.success("✅ Liste mise à jour")
                            
