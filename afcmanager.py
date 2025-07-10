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
    fig.update_yaxes(showticklabels=False, range=[-25, 125], visible=False)

    fig.update_layout(
        width=460,
        height=800,
        plot_bgcolor="#154734",
        margin=dict(l=10, r=10, t=10, b=10),
        showlegend=False
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
                    if s.get("note"):
                        parts.append(f"⭐ {s['note']}")
                    if s.get("hdm"):
                        parts.append("🏆")
                    stats = " | ".join(parts)

                hovertext = f"{nom}{' (C)' if joueur.get('Capitaine') else ''}"
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

                # Stats sous le joueur
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

    # --- 🔁 Affichage des remplaçants ---
    remplacants = remplacants or []
    n = len(remplacants)
    if n:
        if n == 1:
            positions = [(34, -10)]
        elif n == 2:
            positions = [(28, -10), (40, -10)]
        elif n == 3:
            positions = [(22, -10), (34, -10), (46, -10)]
        elif n == 4:
            positions = [(22, -8), (34, -8), (46, -8), (34, -17)]
        else:
            positions = [(18, -8), (34, -8), (50, -8), (26, -17), (42, -17)]

        for idx, remp in enumerate(remplacants[:5]):
            x_r, y_r = positions[idx]
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
                if s.get("note"):
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
    return {poste: [None for _ in range(FORMATION[formation][poste])] for poste in POSTES_ORDER}

# --- 🎮 Interface pour sélectionner les titulaires dynamiquement ---
def terrain_interactif(formation, terrain_key, key_suffix=None):
    if st.session_state.players.empty:
        st.info("Aucun joueur dans la base. Merci d'importer ou d'ajouter des joueurs.")
        return {poste: [] for poste in POSTES_ORDER}

    if terrain_key not in st.session_state:
        st.session_state[terrain_key] = terrain_init(formation)
    terrain = st.session_state[terrain_key]

    stats_data = []
    for _, row in st.session_state.players.iterrows():
        s = compute_player_stats(row["Nom"])
        stats_data.append({**row, **s})
    stats_df = pd.DataFrame(stats_data)

    # Tri des joueurs par titularisations
    stats_df["Titularisations"] = pd.to_numeric(stats_df.get("Titularisations", 0), errors="coerce").fillna(0)
    joueurs_tries = stats_df.sort_values("Titularisations", ascending=False)["Nom"].tolist()

    for poste in POSTES_ORDER:
        noms_postes = POSTES_NOMS.get(formation, {}).get(poste, [])
        if not noms_postes:
            noms_postes = [f"{POSTES_LONG[poste]} {i+1}" for i in range(FORMATION[formation][poste])]
        for i in range(FORMATION[formation][poste]):
            all_selected = [j["Nom"] for p in POSTES_ORDER for j in terrain.get(p, []) if isinstance(j, dict) and j]
            current = terrain[poste][i]
            current_nom = current["Nom"] if current and isinstance(current, dict) else ""
            label = noms_postes[i] if i < len(noms_postes) else f"{POSTES_LONG[poste]} {i+1}"
            options = [""] + [n for n in joueurs_tries if n == current_nom or n not in all_selected]
            key_select = f"{terrain_key}_{poste}_{i}"
            if key_suffix:
                key_select += f"_{key_suffix}"
            choix = st.selectbox(label, options, index=options.index(current_nom) if current_nom in options else 0, key=key_select)
            if choix:
                joueur_info = st.session_state.players[st.session_state.players["Nom"] == choix].iloc[0].to_dict()
                num = st.text_input(f"Numéro de {choix}", value=current.get("Numero", "") if current else "", key=f"num_{terrain_key}_{poste}_{i}")
                cap = st.checkbox(f"Capitaine ?", value=current.get("Capitaine", False) if current else False, key=f"cap_{terrain_key}_{poste}_{i}")
                joueur_info["Numero"] = num
                joueur_info["Capitaine"] = cap
                terrain[poste][i] = joueur_info
            else:
                terrain[poste][i] = None

    st.session_state[terrain_key] = terrain
    return terrain

def remplacants_interactif(key, titulaires, key_suffix=None):
    if f"remp_{key}" not in st.session_state:
        st.session_state[f"remp_{key}"] = [{"Nom": None, "Numero": ""} for _ in range(MAX_REMPLACANTS)]
    remps = st.session_state[f"remp_{key}"]

    stats_data = []
    for _, row in st.session_state.players.iterrows():
        s = compute_player_stats(row["Nom"])
        stats_data.append({**row, **s})
    stats_df = pd.DataFrame(stats_data)

    stats_df["Titularisations"] = pd.to_numeric(stats_df.get("Titularisations", 0), errors="coerce").fillna(0)
    noms_joueurs_tries = stats_df.sort_values("Titularisations", ascending=False)["Nom"].tolist()

    dispo = [n for n in noms_joueurs_tries if n not in titulaires and n not in [r["Nom"] for r in remps if r["Nom"]]]

    for i in range(MAX_REMPLACANTS):
        current = remps[i]["Nom"]
        options = dispo + ([current] if current and current not in dispo else [])
        key_select = f"remp_choice_{key}_{i}"
        if key_suffix:
            key_select += f"_{key_suffix}"
        choix = st.selectbox(
            f"Remplaçant {i+1}",
            [""] + options,
            index=(options.index(current)+1) if current in options else 0,
            key=key_select
        )
        if choix:
            joueur_info = st.session_state.players[st.session_state.players["Nom"] == choix].iloc[0].to_dict()
            num = st.text_input(f"Numéro de {choix}", value=remps[i].get("Numero",""), key=f"num_remp_{key}_{i}")
            remps[i] = {"Nom": choix, "Numero": num}
        else:
            remps[i] = {"Nom": None, "Numero": ""}
        dispo = [n for n in dispo if n != choix]

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
with st.sidebar:
    st.header("🔧 Options")
    if st.button("🔄 Recharger les données depuis GitHub"):
        manager.load()
        st.success("✅ Données rechargées")
        st.experimental_rerun()

# --- 🧭 Onglets principaux de navigation ---
tab_acc, tab1, tab2, tab3, tab4 = st.tabs([
    "🏠", 
    "📅 Matchs", 
    "📈 Championnat", 
    "🧠 Gestion Équipe", 
    "🧪 Tactiques"
])

# --- 🏟️ Onglet Accueil (Tableau de bord) ---
with tab_acc:
    st.title("🏟️ Tableau de bord AFC")

    today = datetime.today().date()
    matchs = st.session_state.get("matchs", {})
    classement = get_classement(
        st.session_state.get("championnat_scores", {}),
        st.session_state.get("adversaires", [])
    )

    # 📊 Classement championnat
    try:
        rang_afc = classement.reset_index(drop=True).query("Équipe == 'AFC'").index[0] + 1
        st.markdown(f"***{rang_afc}ᵉ***")
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
        st.markdown(f"🏆 Coupe : **{match_coupe_a_venir.get('journee', 'Tour à venir')}** ")
    elif dernier_tour_coupe:
        st.markdown(f"🏆 Coupe : **{dernier_tour_coupe.get('journee', 'Tour inconnu')}** ")
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
        st.markdown(" 📈 Forme récente de l'équipe ".join(derniers_resultats))
    else:
        st.info("Aucun match joué cette saison.")

    # 📅 Prochain match
    st.subheader("📅 Prochain match toutes compétitions")
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
        st.markdown(f"**{prochain_match.get('type','')} - {prochain_match.get('journee', '')}**")
        st.markdown(f"🆚 **Adversaire** : {prochain_match['adversaire']}")
        st.markdown(f"📅 **Date** : {prochain_match['date']}")
        lieu = "🏠 Domicile" if prochain_match.get("domicile", True) else "🚗 Extérieur"
        st.markdown(f"📍 **Lieu** : {lieu}")
    else:
        st.info("Aucun match à venir.")


# --- 🧠 Onglet : Gestion Équipe ---
with tab3:
    subtab1, subtab2 = st.tabs(["📊 Stats équipe", "📋 Base joueurs"])

    # -- 🟡 Sous-onglet : Statistiques équipe --
    with subtab1:
        st.title("📊 Statistiques de l'équipe")

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

            col1, col2 = st.columns(2)
            with col1:
                st.subheader("⭐ Top 5 Notes")
                st.dataframe(top_rating[["Nom", "Note générale"]], use_container_width=True, hide_index=True)
                st.subheader("⚽ Top 5 Buteurs")
                st.dataframe(top_buts[["Nom", "Buts"]], use_container_width=True, hide_index=True)
                st.subheader("🎯 Top 5 Passeurs")
                st.dataframe(top_passes[["Nom", "Passes décisives"]], use_container_width=True, hide_index=True)
                st.subheader("🔥 Top 5 Décisifs")
                st.dataframe(top_decisive[["Nom", "Buts + Passes"]], use_container_width=True, hide_index=True)
                st.subheader("🧤 Clean Sheets")
                st.dataframe(top_clean[["Nom", "Clean sheets"]], use_container_width=True, hide_index=True)
            with col2:
                st.subheader("⚡ Ratio par match")
                st.dataframe(top_ratio[["Nom", "Décisif par match"]], use_container_width=True, hide_index=True)
                st.subheader("🔁 Plus utilisés")
                st.dataframe(top_used[["Nom", "Titularisations"]], use_container_width=True, hide_index=True)
                st.subheader("🟥🟨 Bouchers")
                st.dataframe(top_bouchers[["Nom", "Cartons rouges", "Cartons jaunes"]], use_container_width=True, hide_index=True)

            # 🏆 Statistiques globales de l'équipe
            st.markdown("---")
            col3, col4, col5 = st.columns(3)
            st.metric("🧮 Buts marqués", int(df["Buts"].sum()))
            st.metric("🔓 Buts encaissés", sum(m.get("score_adv", 0) for m in st.session_state.matchs.values() if m.get("noted")))
            st.metric("👥 Nombre de buteurs", df[df["Buts"] > 0]["Nom"].nunique())
        else:
            st.info("Aucun joueur dans la base pour générer des stats.")

    # -- 🟠 Sous-onglet : Base de données joueurs --
    with subtab2:
        st.title("📋 Base de données joueurs")
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

        if st.button("💾 Sauvegarder les modifications", key="btn_save_joueurs"):
            edited_df = edited_df.fillna("")
            edited_df = edited_df[edited_df["Nom"].str.strip() != ""]
            st.session_state.players = edited_df[PLAYER_COLS]
            manager.save()
            st.success("Base de joueurs mise à jour ✅")
            st.rerun()
        st.caption("🗑️ Pour supprimer un joueur, videz son nom et cliquez sur Sauvegarder.")

# --- 🧪 Onglet Tactiques ---
with tab4:
    st.title("🧪 Gestion des tactiques et compositions")

    subtab1, subtab2, subtab3 = st.tabs([
        "📐 Créer une composition", 
        "🗂️ Mes compositions", 
        "🔎 Profondeur d'effectif"
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
            st.session_state["terrain_create_compo"] = loaded["details"]
            del st.session_state[edit_key]

        nom_compo = st.text_input("📝 Nom de la composition", key="nom_compo_create", value=nom_compo if edit_compo else "")
        formation = st.selectbox("📌 Formation", list(FORMATION.keys()), index=list(FORMATION.keys()).index(st.session_state.get("formation_create_compo", DEFAULT_FORMATION)), key="formation_create_compo")

        col_left, col_right = st.columns([1, 2])
        with col_left:
            terrain = terrain_interactif(formation, "terrain_create_compo")
            titulaires = [j["Nom"] for p in POSTES_ORDER for j in terrain.get(p, []) if j]
            remplacants = remplacants_interactif("create_compo", titulaires)

        with col_right:
            fig = draw_football_pitch_vertical()
            fig = plot_lineup_on_pitch_vertical(fig, terrain, formation, remplacants)
            st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key="fig_create_compo")

        # Sauvegarde
        if st.button("💾 Sauvegarder la composition"):
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
                    if col1.button(f"✏️ Modifier {nom}", key=f"edit_{nom}"):
                        st.session_state["edit_compo"] = (nom, compo)
                        st.rerun()
                    if col2.button(f"🗑️ Supprimer {nom}", key=f"delete_{nom}"):
                        del st.session_state.lineups[nom]
                        manager.save()
                        st.success("✅ Composition supprimée")
                        st.rerun()

    # --- 🔎 Profondeur d'effectif par poste ---
    with subtab3:
        st.title("🔍 Profondeur d'effectif")
        formation_selected = st.selectbox("🎯 Formation", list(FORMATION.keys()), key="formation_profondeur")

        # 🔍 Affichage récapitulatif de la profondeur existante
        st.markdown("#### 📋 Profondeur enregistrée")
        prof = st.session_state.profondeur_effectif.get(formation_selected, {})
        for poste in POSTES_ORDER:
            if poste in POSTES_NOMS.get(formation_selected, {}):
                for idx, label in enumerate(POSTES_NOMS[formation_selected][poste]):
                    options = prof.get(poste, {}).get(idx, [])
                    if options:
                        st.markdown(f"- **{label}** : {', '.join(options)}")

        if formation_selected not in st.session_state.profondeur_effectif:
            st.session_state.profondeur_effectif[formation_selected] = {}

        profondeur = st.session_state.profondeur_effectif[formation_selected]
        joueurs = st.session_state.players["Nom"].dropna().tolist()

        col_left, col_right = st.columns([3, 7])
        with col_left:
            st.markdown("### Sélection par poste")
            postes_formation = POSTES_NOMS[formation_selected]
            for poste in POSTES_ORDER:
                if poste not in postes_formation:
                    continue
                if poste not in profondeur:
                    profondeur[poste] = {}

                for idx_label, label in enumerate(postes_formation[poste]):
                    key_poste = f"{formation_selected}_{poste}_{idx_label}"
                    noms = profondeur[poste].get(idx_label, [])
                    noms = noms if isinstance(noms, list) else []
                    while len(noms) < 1 or (noms and noms[-1].strip()):
                        noms.append("")
                    noms = [n.strip() for n in noms if isinstance(n, str)]

                    st.markdown(f"**{label}**")
                    for i in range(len(noms)):
                        select_key = f"{key_poste}_choix_{i}"
                        choix = st.selectbox(f"Option {i+1}", [""] + joueurs, index=([""] + joueurs).index(noms[i]) if noms[i] in joueurs else 0, key=select_key)
                        noms[i] = choix

                    # Nettoyage
                    noms = [n for n in noms if n.strip()]
                    profondeur[poste][idx_label] = noms

            if st.button("💾 Sauvegarder profondeur"):
                st.session_state.profondeur_effectif[formation_selected] = profondeur
                manager.save()
                st.success("✅ Profondeur sauvegardée")

        with col_right:
            st.markdown("### Visualisation du terrain")
            fig = draw_football_pitch_vertical()
            positions = positions_for_formation_vertical(formation_selected)
            for poste in POSTES_ORDER:
                for idx, label in enumerate(POSTES_NOMS[formation_selected].get(poste, [])):
                    noms = profondeur.get(poste, {}).get(idx, [])
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
    st.title("📅 Gestion des matchs")
    subtab1, subtab2 = st.tabs(["⚙️ Créer un match", "📋 Mes matchs"])

    # --- ⚙️ Créer un match ---
    with subtab1:
        st.markdown("### Paramètres du match")

        type_match = st.selectbox("🧭 Type de match", ["Championnat", "Coupe", "Amical"])
        if type_match=="Championnat":
            journee = st.text_input("📌 Journée", value="J")
        else:
            if type_match=="Coupe":
                journee=st.selectbox("📌 Tour", ["Poules", "Huitièmes", "Quarts", "Demies", "Finale"])
            else:
                journee = st.text_input("📌 Numéro", value="#")
        adversaires_list = st.session_state.get("adversaires", [])
        adversaire_select = st.selectbox("👥 Adversaire", adversaires_list + ["Autre..."])
        if adversaire_select == "Autre...":
            adversaire = st.text_input("🆕 Nom de l'adversaire")
        else:
            adversaire = adversaire_select

        date = st.date_input("📅 Date du match", value=datetime.today())
        heure = st.time_input("🕒 Heure du match", value=datetime.strptime("21:00", "%H:%M").time())
        domicile = st.radio("📍 Lieu du match", ["Domicile", "Extérieur"])
        if domicile == "Domicile":
            lieu_default = "Club de Football Barradels, 2 Rue des Cyclamens, 31700 Blagnac"
        else:
            lieu_default = ""
        
        lieu = st.text_input("📌 Lieu du match", value=lieu_default)

        nom_match = f"{type_match} - {journee} - {'AFC vs' if domicile == 'Domicile' else ''} {adversaire}{' vs AFC' if domicile == 'Extérieur' else ''}"

        if st.button("✅ Enregistrer le match"):
            match_id = str(uuid.uuid4())
            st.session_state.matchs[match_id] = {
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
            manager.save()
            st.success("📦 Match enregistré")
            st.rerun()

    # --- 📋 Mes matchs enregistrés ---
    with subtab2:
        if not st.session_state.matchs:
            st.info("📭 Aucun match enregistré.")
        else:
            for mid, match in st.session_state.matchs.items():
                with st.expander(match.get("nom_match", "Match sans nom")):
                    
                    # --- ✅ Checkbox “Match terminé” ---
                    match_ended = st.checkbox("Match terminé", value=match.get("termine", False), key=f"ended_{mid}")
                    if match_ended != match.get("termine", False):
                        match["termine"] = match_ended
                        st.session_state.matchs[mid] = match
                        manager.save()
                        st.rerun()

                    # --- 🏟️ Créer composition pour ce match ---
                    if not match.get("termine"):
                        with st.expander("### 🏟️ Composition du match"):

                            use_compo = st.checkbox("🔁 Utiliser une compo enregistrée ?", key=f"use_compo_{mid}")
                            if use_compo and st.session_state.lineups:
                                compo_choice = st.selectbox("📂 Choisir une compo", list(st.session_state.lineups.keys()), key=f"compo_choice_{mid}")
                                compo_data = st.session_state.lineups[compo_choice]
                                formation = compo_data["formation"]
                                terrain = compo_data["details"]
                                remplacants = compo_data.get("remplacants", [])
                            else:
                                formation = st.selectbox("📌 Formation", list(FORMATION.keys()), key=f"form_{mid}")
                                terrain = terrain_interactif(formation, f"terrain_match_{mid}", key_suffix=mid)
                                titulaires = [j["Nom"] for p in POSTES_ORDER for j in terrain.get(p, []) if j]
                                remplacants = remplacants_interactif(f"match_{mid}", titulaires, key_suffix=mid)
    
                            fig = draw_football_pitch_vertical()
                            fig = plot_lineup_on_pitch_vertical(fig, terrain, formation, remplacants)
                            st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"fig_match_{mid}")
    
                            if st.button("💾 Valider la composition", key=f"btn_compo_{mid}"):
                                match["formation"] = formation
                                match["details"] = terrain
                                match["remplacants"] = remplacants
                                st.session_state.matchs[mid] = match
                                manager.save()
                                st.success("✅ Composition enregistrée")
                                st.rerun()

                        # --- 👥 Convocation des joueurs ---
                        if terrain:
                            with st.expander("### 👥 Convocation des joueurs"):
                            
                                joueurs_convoques = []
                                for p in POSTES_ORDER:
                                    joueurs_convoques += [j["Nom"] for j in terrain.get(p, []) if j]
                                joueurs_convoques += [r["Nom"] for r in remplacants if r.get("Nom")]
                                joueurs_convoques = list(dict.fromkeys(joueurs_convoques))
    
                                # Heure de RDV
                                heure_match = match.get("heure", "21:00")
                                try:
                                    rdv = (datetime.strptime(heure_match, "%H:%M") - timedelta(hours=1)).strftime("%H:%M")
                                except:
                                    rdv = "?"
    
                                st.write(f"📅 {match['date']} à {heure_match} – **RDV : {rdv}**")
                                st.write(f"{match.get('adversaire')}")
                                st.write(f"📍 Lieu : {match.get('lieu')}")
                                for nom in joueurs_convoques:
                                    st.markdown(f"- {nom}")

                    # --- 📝 Saisie des statistiques du match ---
                    elif match_ended and not match.get("noted", False):
                        with st.expander("### 📝 Statistiques du match"):

                            joueurs = [j["Nom"] for p in POSTES_ORDER for j in match.get("details", {}).get(p, []) if j]
                            joueurs += [r["Nom"] for r in match.get("remplacants", []) if r.get("Nom")]
                            joueurs = list(dict.fromkeys(joueurs))
    
                            score_afc = st.number_input("⚽ Buts AFC", min_value=0, max_value=20, value=0, key=f"score_afc_{mid}")
                            score_adv = st.number_input(f"⚽ Buts {match['adversaire']}", min_value=0, max_value=20, value=0, key=f"score_adv_{mid}")
    
                            events = {
                                "buteurs": {},
                                "passeurs": {},
                                "cartons_jaunes": {},
                                "cartons_rouges": {},
                                "notes": {}
                            }
    
                            for nom in joueurs:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    events["buteurs"][nom] = st.number_input(f"{nom} - Buts", min_value=0, value=0, key=f"but_{mid}_{nom}")
                                with col2:
                                    events["passeurs"][nom] = st.number_input(f"{nom} - Passes", min_value=0, value=0, key=f"pass_{mid}_{nom}")
                                with col3:
                                    events["notes"][nom] = st.slider(f"{nom} - Note", min_value=0.0, max_value=10.0, value=5.0, step=0.5, key=f"note_{mid}_{nom}")
                                events["cartons_jaunes"][nom] = st.number_input(f"{nom} - 🟨 Jaunes", min_value=0, value=0, key=f"cj_{mid}_{nom}")
                                events["cartons_rouges"][nom] = st.number_input(f"{nom} - 🟥 Rouges", min_value=0, value=0, key=f"cr_{mid}_{nom}")
    
                            hdm = st.selectbox("🏆 Homme du match", [""] + joueurs, key=f"hdm_{mid}")
    
                            if st.button("✅ Valider les stats", key=f"valide_{mid}"):
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
                            st.markdown(f"**{match['nom_match']}**")
                            
                            # Score
                            if match.get("domicile") == "Domicile":
                                st.markdown(f"### AFC {match['score_afc']} - {match['score_adv']} {match['adversaire']}")
                            else:
                                st.markdown(f"### {match['adversaire']} {match['score_adv']} - {match['score_afc']} AFC")

                            hdm = match.get("homme_du_match")
                            if hdm:
                                st.markdown(f"**🏆 Homme du match :** {hdm}")

                            st.markdown("---")
                            
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
                                    "note": None,
                                    "hdm": match.get("homme_du_match") == nom
                                }
                                for nom in joueurs_all
                            }
                            
                            fig = plot_lineup_on_pitch_vertical(
                                fig,
                                match["details"],
                                match["formation"],
                                match.get("remplacants", []),
                                player_stats=player_stats
                            )
                            
                            st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True}, key=f"fig_match_{mid}")
                            if st.button("✏️ Modifier les statistiques", key=f"edit_stats_{mid}"):
                                match["noted"] = False
                                st.session_state.matchs[mid] = match
                                manager.save()
                                st.rerun()
                            
                    if st.button("🗑️ Supprimer ce match", key=f"delete_match_{mid}"):
                        del st.session_state.matchs[mid]
                        manager.save()
                        st.success("🧹 Match supprimé")
                        st.rerun()

# --- 📈 Onglet Suivi Championnat ---
with tab2:
    subtab1, subtab2, subtab3 = st.tabs([
        "🏆 Classement", 
        "📋 Saisie des scores", 
        "🧑‍🤝‍🧑 Gestion des adversaires"
    ])

    # --- 🏆 Classement automatique ---
    with subtab1:
        st.title("🏆 Classement du championnat")

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
        st.title("📋 Saisie des scores")

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
            col1, col2, col3 = st.columns([1, 4, 1])
            with col1:
                if idx > 0 and st.button("←", key="journee_prev"):
                    st.session_state.selected_journee = journees[idx - 1]
                    st.rerun()
            with col2:
                st.markdown(f"### 📅 Journée : {st.session_state.selected_journee}")
            with col3:
                if idx < len(journees) - 1 and st.button("→", key="journee_next"):
                    st.session_state.selected_journee = journees[idx + 1]
                    st.rerun()

        selected = st.session_state.selected_journee
        matchs = st.session_state.championnat_scores.get(selected, [])

        # Affichage/édition des matchs
        for i, match in enumerate(matchs):
            col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 3])
            dom = col1.selectbox(f"🏠 Domicile {i+1}", equipes, index=equipes.index(match["domicile"]), key=f"dom_{selected}_{i}")
            score_dom = col2.number_input("⚽", value=match.get("score_dom", 0), min_value=0, max_value=30, key=f"score_dom_{selected}_{i}")
            score_ext = col4.number_input("⚽", value=match.get("score_ext", 0), min_value=0, max_value=30, key=f"score_ext_{selected}_{i}")
            ext = col5.selectbox(f"🚗 Extérieur {i+1}", equipes, index=equipes.index(match["exterieur"]), key=f"ext_{selected}_{i}")
            match.update({"domicile": dom, "score_dom": score_dom, "exterieur": ext, "score_ext": score_ext})

        # Ajouter un match
        st.markdown("---")
        st.markdown("### ➕ Ajouter un match à cette journée")
        with st.form(f"add_match_form_{selected}"):
            dom_new = st.selectbox("🏠 Équipe à domicile", equipes)
            ext_new = st.selectbox("🚗 Équipe à l'extérieur", [e for e in equipes if e != dom_new])
            score_dom_new = st.number_input("⚽ Score domicile", min_value=0, value=0)
            score_ext_new = st.number_input("⚽ Score extérieur", min_value=0, value=0)
            if st.form_submit_button("📦 Ajouter"):
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

        # Ajouter une nouvelle journée
        if st.button("🗓️ Ajouter une journée"):
            new_key = next_journee_key()
            st.session_state.championnat_scores[new_key] = []
            manager.save()
            st.success(f"📅 {new_key} créée")
            st.rerun()

    # --- 🧑‍🤝‍🧑 Gestion des adversaires ---
    with subtab3:
        st.title("🧑‍🤝‍🧑 Adversaires du championnat")

        adv_df = pd.DataFrame({"Nom": st.session_state.adversaires}, dtype="object")
        edited_df = st.data_editor(adv_df, num_rows="dynamic", hide_index=True)

        if st.button("💾 Sauvegarder les adversaires"):
            st.session_state.adversaires = edited_df["Nom"].dropna().astype(str).tolist()
            manager.save()
            st.success("✅ Liste mise à jour")
                            
