import streamlit as st
import pandas as pd
import json
import plotly.graph_objects as go
from datetime import datetime
import os

# Configuration and Page Setup
st.set_page_config(
    page_title="⚽ AFC Team Manager",
    page_icon="⚽",
    layout="wide"
)

# Initialize session state for data persistence
if 'data' not in st.session_state:
    # Initialize empty data structure
    st.session_state.data = {
        'players': [],
        'lineups': [],
        'matches': []
    }
    
    # Try to load existing data
    if os.path.exists('afcdata.json'):
        try:
            with open('afcdata.json', 'r', encoding='utf-8') as f:
                st.session_state.data = json.load(f)
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")

# Sidebar
st.sidebar.title("⚽ Gestion d'Équipe AFC")

# Data Import/Export in Sidebar
st.sidebar.header("Import/Export Data")

# Export Data
if st.sidebar.button("Export Data (JSON)"):
    json_str = json.dumps(st.session_state.data, indent=2, ensure_ascii=False)
    b64 = base64.b64encode(json_str.encode()).decode()
    href = f'<a href="data:file/json;base64,{b64}" download="afcdata.json">Download JSON</a>'
    st.sidebar.markdown(href, unsafe_allow_html=True)

# Import Data
uploaded_file = st.sidebar.file_uploader("Import JSON Data", type=['json'])
if uploaded_file is not None:
    try:
        imported_data = json.load(uploaded_file)
        # Validate data structure
        required_keys = ['players', 'lineups', 'matches']
        if all(key in imported_data for key in required_keys):
            st.session_state.data = imported_data
            st.sidebar.success("Data imported successfully!")
        else:
            st.sidebar.error("Invalid data structure in JSON file")
    except Exception as e:
        st.sidebar.error(f"Error importing data: {str(e)}")

# Main Tabs
tab1, tab2, tab3 = st.tabs(["Base joueurs", "Compositions", "Matchs"])

with tab1:
    st.header("Base joueurs")
    
    # Convert players list to DataFrame for editing
    if not st.session_state.data['players']:
        df_players = pd.DataFrame(columns=['Name', 'Position', 'Info'])
    else:
        df_players = pd.DataFrame(st.session_state.data['players'])
    
    # Calculate player stats
    def calculate_player_stats(player_name):
        stats = {
            'goals': 0, 'assists': 0, 'yellow_cards': 0, 'red_cards': 0,
            'selections': 0, 'starts': 0, 'ratings': [], 'motm': 0
        }
        
        for match in st.session_state.data['matches']:
            if match.get('completed', False):
                # Count selections and starts
                if player_name in match['lineup'] + match.get('substitutes', []):
                    stats['selections'] += 1
                    if player_name in match['lineup']:
                        stats['starts'] += 1
                
                # Count goals, assists, cards
                for goal in match.get('goals', []):
                    if goal['scorer'] == player_name:
                        stats['goals'] += 1
                    if goal['assist'] == player_name:
                        stats['assists'] += 1
                
                for card in match.get('cards', []):
                    if card['player'] == player_name:
                        if card['type'] == 'yellow':
                            stats['yellow_cards'] += 1
                        elif card['type'] == 'red':
                            stats['red_cards'] += 1
                
                # Ratings and MOTM
                if match.get('ratings', {}).get(player_name):
                    stats['ratings'].append(match['ratings'][player_name])
                if match.get('motm') == player_name:
                    stats['motm'] += 1
        
        # Calculate averages and per-match stats
        stats['avg_rating'] = sum(stats['ratings']) / len(stats['ratings']) if stats['ratings'] else 0
        stats['decisions_per_match'] = (stats['goals'] + stats['assists']) / stats['selections'] if stats['selections'] > 0 else 0
        
        return stats

    # Editable player table
    edited_df = st.data_editor(
        df_players,
        use_container_width=True,
        num_rows="dynamic",
        key="player_editor"
    )
    
    # Save button for player changes
    if st.button("Sauvegarder les modifications"):
        st.session_state.data['players'] = edited_df.to_dict('records')
        with open('afcdata.json', 'w', encoding='utf-8') as f:
            json.dump(st.session_state.data, f, indent=2, ensure_ascii=False)
        st.success("Modifications sauvegardées!")

    # Display player stats
    if st.session_state.data['players']:
        st.subheader("Statistiques des joueurs")
        stats_list = []
        for player in st.session_state.data['players']:
            stats = calculate_player_stats(player['Name'])
            stats_list.append({
                'Name': player['Name'],
                'Position': player['Position'],
                'Goals': stats['goals'],
                'Assists': stats['assists'],
                'G+A': stats['goals'] + stats['assists'],
                'G+A per match': round(stats['decisions_per_match'], 2),
                'Yellow Cards': stats['yellow_cards'],
                'Red Cards': stats['red_cards'],
                'Selections': stats['selections'],
                'Starts': stats['starts'],
                'Avg Rating': round(stats['avg_rating'], 1),
                'MOTM': stats['motm']
            })
        
        st.dataframe(
            pd.DataFrame(stats_list),
            use_container_width=True
        )

def draw_football_pitch(formation):
    # Define pitch dimensions and colors
    pitch = go.Figure()
    
    # Draw pitch lines
    pitch.add_shape(type="rect", x0=0, y0=0, x1=100, y1=70,
                   line=dict(color="white", width=2),
                   fillcolor="darkgreen")
    
    # Add penalty areas, center circle, etc.
    # Penalty areas
    pitch.add_shape(type="rect", x0=0, y0=20, x1=16, y1=50,
                   line=dict(color="white", width=2), fillcolor="darkgreen")
    pitch.add_shape(type="rect", x0=84, y0=20, x1=100, y1=50,
                   line=dict(color="white", width=2), fillcolor="darkgreen")
    
    # Center line and circle
    pitch.add_shape(type="line", x0=50, y0=0, x1=50, y1=70,
                   line=dict(color="white", width=2))
    pitch.add_shape(type="circle", x0=45, y0=30, x1=55, y1=40,
                   line=dict(color="white", width=2))
    
    # Update layout
    pitch.update_layout(
        showlegend=False,
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        plot_bgcolor='darkgreen',
        paper_bgcolor='darkgreen',
        margin=dict(l=0, r=0, t=0, b=0),
        height=600
    )
    
    return pitch

# Formation positions dictionary
FORMATIONS = {
    "4-4-2": {
        "positions": ["GK", "LB", "CB", "CB", "RB", "LM", "CM", "CM", "RM", "ST", "ST"],
        "coordinates": [(5, 35),  # GK
                       (20, 15), (20, 28), (20, 42), (20, 55),  # Defenders
                       (40, 15), (40, 28), (40, 42), (40, 55),  # Midfielders
                       (60, 28), (60, 42)]  # Strikers
    },
    "4-2-3-1": {
        "positions": ["GK", "LB", "CB", "CB", "RB", "CDM", "CDM", "LW", "CAM", "RW", "ST"],
        "coordinates": [(5, 35),
                       (20, 15), (20, 28), (20, 42), (20, 55),
                       (35, 28), (35, 42),
                       (50, 15), (50, 35), (50, 55),
                       (65, 35)]
    },
    "4-3-3": {
        "positions": ["GK", "LB", "CB", "CB", "RB", "CM", "CM", "CM", "LW", "ST", "RW"],
        "coordinates": [(5, 35),
                       (20, 15), (20, 28), (20, 42), (20, 55),
                       (40, 25), (40, 35), (40, 45),
                       (60, 15), (60, 35), (60, 55)]
    }
}

with tab2:
    tab2_1, tab2_2 = st.tabs(["Créer une composition", "Mes compositions"])
    
    with tab2_1:
        st.header("Créer une composition")
        
        # Lineup name
        lineup_name = st.text_input("Nom de la composition")
        
        # Formation selector
        formation = st.selectbox("Formation", list(FORMATIONS.keys()))
        
        # Get available players
        available_players = [p['Name'] for p in st.session_state.data['players']]
        available_players.insert(0, "")  # Empty option for clearing positions
        
        # Create columns for position selection
        positions = FORMATIONS[formation]["positions"]
        coordinates = FORMATIONS[formation]["coordinates"]
        
        # Initialize lineup dict
        lineup_dict = {
            "name": lineup_name,
            "formation": formation,
            "players": {},
            "numbers": {},
            "substitutes": [],
            "captain": None
        }
        
        # Create position selectors
        st.subheader("Starting Lineup")
        cols = st.columns(3)
        for idx, (pos, coord) in enumerate(zip(positions, coordinates)):
            with cols[idx % 3]:
                selected_player = st.selectbox(
                    f"{pos}",
                    options=[p for p in available_players if p not in lineup_dict["players"].values()],
                    key=f"pos_{idx}"
                )
                if selected_player:
                    lineup_dict["players"][pos] = selected_player
                    lineup_dict["numbers"][selected_player] = st.number_input(
                        f"Number for {selected_player}",
                        min_value=1,
                        max_value=99,
                        value=idx + 1,
                        key=f"num_{idx}"
                    )
        
        # Substitutes
        st.subheader("Remplaçants")
        sub_cols = st.columns(5)
        for i in range(5):
            with sub_cols[i]:
                sub = st.selectbox(
                    f"Substitute {i+1}",
                    options=[p for p in available_players if p not in lineup_dict["players"].values() 
                            and p not in lineup_dict["substitutes"]],
                    key=f"sub_{i}"
                )
                if sub:
                    lineup_dict["substitutes"].append(sub)
                    lineup_dict["numbers"][sub] = st.number_input(
                        f"Number for {sub}",
                        min_value=1,
                        max_value=99,
                        value=12+i,
                        key=f"sub_num_{i}"
                    )
        
        # Captain selection
        captain_options = [p for p in lineup_dict["players"].values() if p]
        if captain_options:
            lineup_dict["captain"] = st.selectbox("Captain", options=captain_options)
        
        # Draw the lineup
        if lineup_dict["players"]:
            pitch = draw_football_pitch(formation)
            
            # Add players to pitch
            for pos, coord in zip(positions, coordinates):
                player = lineup_dict["players"].get(pos, "")
                if player:
                    text = f"{player} ({lineup_dict['numbers'][player]})"
                    if player == lineup_dict["captain"]:
                        text += " ©️"
                    pitch.add_annotation(
                        x=coord[0], y=coord[1],
                        text=text,
                        showarrow=False,
                        font=dict(color="white"),
                        bgcolor="rgba(0,0,139,0.7)"
                    )
            
            st.plotly_chart(pitch, use_container_width=True)
        
        # Save lineup
        if st.button("Sauvegarder la composition"):
            if not lineup_name:
                st.error("Please enter a lineup name")
            elif len(lineup_dict["players"]) < len(positions):
                st.error("Please fill all positions")
            else:
                st.session_state.data["lineups"].append(lineup_dict)
                with open('afcdata.json', 'w', encoding='utf-8') as f:
                    json.dump(st.session_state.data, f, indent=2, ensure_ascii=False)
                st.success("Composition saved successfully!")
    
    with tab2_2:
        st.header("Mes compositions")
        
        for idx, lineup in enumerate(st.session_state.data["lineups"]):
            with st.expander(f"{lineup['name']} ({lineup['formation']})"):
                # Display lineup visualization
                pitch = draw_football_pitch(lineup['formation'])
                positions = FORMATIONS[lineup['formation']]["positions"]
                coordinates = FORMATIONS[lineup['formation']]["coordinates"]
                
                for pos, coord in zip(positions, coordinates):
                    player = lineup["players"].get(pos, "")
                    if player:
                        text = f"{player} ({lineup['numbers'][player]})"
                        if player == lineup["captain"]:
                            text += " ©️"
                        pitch.add_annotation(
                            x=coord[0], y=coord[1],
                            text=text,
                            showarrow=False,
                            font=dict(color="white"),
                            bgcolor="rgba(0,0,139,0.7)"
                        )
                
                st.plotly_chart(pitch, use_container_width=True)
                
                # Display substitutes
                if lineup["substitutes"]:
                    st.subheader("Substitutes")
                    for sub in lineup["substitutes"]:
                        st.write(f"{sub} ({lineup['numbers'][sub]})")
                
                # Edit and Delete buttons
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Edit", key=f"edit_{idx}"):
                        st.session_state.lineup_to_edit = idx
                with col2:
                    if st.button("Delete", key=f"delete_{idx}"):
                        st.session_state.data["lineups"].pop(idx)
                        with open('afcdata.json', 'w', encoding='utf-8') as f:
                            json.dump(st.session_state.data, f, indent=2, ensure_ascii=False)
                        st.rerun()

with tab3:
    tab3_1, tab3_2 = st.tabs(["Créer un match", "Mes matchs"])
    
    with tab3_1:
        st.header("Créer un match")
        
        # Match details form
        col1, col2 = st.columns(2)
        with col1:
            match_type = st.selectbox("Type", ["Championship", "Cup"])
            opponent = st.text_input("Opponent")
            match_date = st.date_input("Date")
        with col2:
            match_time = st.time_input("Time")
            location = st.text_input("Location")
        
        # Auto-generate match name
        match_name = f"AFC vs {opponent} - {match_date.strftime('%Y-%m-%d')}"
        st.text(f"Match Name: {match_name}")
        
        # Option to load from saved lineup
        st.subheader("Load from saved lineup")
        saved_lineups = [lineup["name"] for lineup in st.session_state.data["lineups"]]
        selected_lineup = st.selectbox("Select lineup", [""] + saved_lineups)
        
        if selected_lineup:
            lineup_data = next(l for l in st.session_state.data["lineups"] if l["name"] == selected_lineup)
            formation = lineup_data["formation"]
            initial_players = lineup_data["players"]
            initial_subs = lineup_data["substitutes"]
            initial_captain = lineup_data["captain"]
            initial_numbers = lineup_data["numbers"]
        else:
            formation = st.selectbox("Formation", list(FORMATIONS.keys()))
            initial_players = {}
            initial_subs = []
            initial_captain = None
            initial_numbers = {}
        
        # Player selection interface (similar to lineup creator but with initial values)
        match_dict = {
            "name": match_name,
            "type": match_type,
            "opponent": opponent,
            "date": match_date.isoformat(),
            "time": match_time.strftime("%H:%M"),
            "location": location,
            "formation": formation,
            "players": {},
            "numbers": {},
            "substitutes": [],
            "captain": None,
            "completed": False,
            "score": {"AFC": 0, "opponent": 0},
            "goals": [],
            "assists": [],
            "cards": [],
            "ratings": {},
            "motm": None
        }
        
        # Position selection (pre-filled if lineup was selected)
        st.subheader("Starting Lineup")
        positions = FORMATIONS[formation]["positions"]
        coordinates = FORMATIONS[formation]["coordinates"]
        
        cols = st.columns(3)
        available_players = [p["Name"] for p in st.session_state.data["players"]]
        available_players.insert(0, "")
        
        for idx, (pos, coord) in enumerate(zip(positions, coordinates)):
            with cols[idx % 3]:
                initial_player = initial_players.get(pos, "")
                selected_player = st.selectbox(
                    f"{pos}",
                    options=[p for p in available_players if p not in match_dict["players"].values() or p == initial_player],
                    key=f"match_pos_{idx}",
                    index=available_players.index(initial_player) if initial_player in available_players else 0
                )
                if selected_player:
                    match_dict["players"][pos] = selected_player
                    initial_number = initial_numbers.get(selected_player, idx + 1)
                    match_dict["numbers"][selected_player] = st.number_input(
                        f"Number for {selected_player}",
                        min_value=1,
                        max_value=99,
                        value=initial_number,
                        key=f"match_num_{idx}"
                    )
        
        # Substitutes (pre-filled if lineup was selected)
        st.subheader("Remplaçants")
        sub_cols = st.columns(5)
        for i in range(5):
            with sub_cols[i]:
                initial_sub = initial_subs[i] if i < len(initial_subs) else ""
                sub = st.selectbox(
                    f"Substitute {i+1}",
                    options=[p for p in available_players if p not in match_dict["players"].values() 
                            and p not in match_dict["substitutes"] or p == initial_sub],
                    key=f"match_sub_{i}",
                    index=available_players.index(initial_sub) if initial_sub in available_players else 0
                )
                if sub:
                    match_dict["substitutes"].append(sub)
                    initial_number = initial_numbers.get(sub, 12+i)
                    match_dict["numbers"][sub] = st.number_input(
                        f"Number for {sub}",
                        min_value=1,
                        max_value=99,
                        value=initial_number,
                        key=f"match_sub_num_{i}"
                    )
        
        # Captain selection
        captain_options = [p for p in match_dict["players"].values() if p]
        if captain_options:
            initial_captain_index = captain_options.index(initial_captain) if initial_captain in captain_options else 0
            match_dict["captain"] = st.selectbox(
                "Captain",
                options=captain_options,
                index=initial_captain_index
            )
        
        # Visualization
        if match_dict["players"]:
            pitch = draw_football_pitch(formation)
            for pos, coord in zip(positions, coordinates):
                player = match_dict["players"].get(pos, "")
                if player:
                    text = f"{player} ({match_dict['numbers'][player]})"
                    if player == match_dict["captain"]:
                        text += " ©️"
                    pitch.add_annotation(
                        x=coord[0], y=coord[1],
                        text=text,
                        showarrow=False,
                        font=dict(color="white"),
                        bgcolor="rgba(0,0,139,0.7)"
                    )
            st.plotly_chart(pitch, use_container_width=True)
        
        # Save match
        if st.button("Sauvegarder le match"):
            if not opponent:
                st.error("Please enter opponent name")
            elif len(match_dict["players"]) < len(positions):
                st.error("Please fill all positions")
            else:
                st.session_state.data["matches"].append(match_dict)
                with open('afcdata.json', 'w', encoding='utf-8') as f:
                    json.dump(st.session_state.data, f, indent=2, ensure_ascii=False)
                st.success("Match saved successfully!")

    with tab3_2:
        st.header("Mes matchs")
        
        for idx, match in enumerate(st.session_state.data["matches"]):
            with st.expander(f"{match['name']} ({match['type']})"):
                if not match["completed"]:
                    # Match completion form
                    col1, col2 = st.columns(2)
                    with col1:
                        afc_score = st.number_input("AFC Score", min_value=0, value=match["score"]["AFC"])
                    with col2:
                        opp_score = st.number_input(f"{match['opponent']} Score", min_value=0, 
                                                  value=match["score"]["opponent"])
                    
                    # Goals and assists
                    st.subheader("Goals")
                    all_players = list(match["players"].values()) + match["substitutes"]
                    
                    for i in range(max(afc_score, len(match["goals"]))):
                        col1, col2 = st.columns(2)
                        with col1:
                            scorer = st.selectbox(
                                f"Goal {i+1} Scorer",
                                options=[""] + all_players,
                                key=f"scorer_{idx}_{i}",
                                index=all_players.index(match["goals"][i]["scorer"])+1 if i < len(match["goals"]) else 0
                            )
                        with col2:
                            assist = st.selectbox(
                                f"Goal {i+1} Assist",
                                options=[""] + all_players,
                                key=f"assist_{idx}_{i}",
                                index=all_players.index(match["goals"][i]["assist"])+1 if i < len(match["goals"]) else 0
                            )
                        if scorer:
                            if i < len(match["goals"]):
                                match["goals"][i] = {"scorer": scorer, "assist": assist if assist else None}
                            else:
                                match["goals"].append({"scorer": scorer, "assist": assist if assist else None})
                    
                    # Cards
                    st.subheader("Cards")
                    card_cols = st.columns(2)
                    with card_cols[0]:
                        if st.button("Add Yellow Card", key=f"add_yellow_{idx}"):
                            match["cards"].append({"type": "yellow", "player": None})
                    with card_cols[1]:
                        if st.button("Add Red Card", key=f"add_red_{idx}"):
                            match["cards"].append({"type": "red", "player": None})
                    
                    for i, card in enumerate(match["cards"]):
                        card["player"] = st.selectbox(
                            f"{card['type'].title()} Card",
                            options=[""] + all_players,
                            key=f"card_{idx}_{i}",
                            index=all_players.index(card["player"])+1 if card["player"] else 0
                        )
                    
                    # Player ratings
                    st.subheader("Player Ratings")
                    rating_cols = st.columns(3)
                    for i, player in enumerate(all_players):
                        with rating_cols[i % 3]:
                            match["ratings"][player] = st.slider(
                                f"{player} Rating",
                                min_value=0.0,
                                max_value=10.0,
                                value=match["ratings"].get(player, 5.0),
                                step=0.5,
                                key=f"rating_{idx}_{i}"
                            )
                    
                    # Man of the Match
                    match["motm"] = st.selectbox(
                        "Man of the Match",
                        options=[""] + all_players,
                        index=all_players.index(match["motm"])+1 if match["motm"] else 0,
                        key=f"motm_{idx}"
                    )
                    
                    # Update match status
                    match["score"]["AFC"] = afc_score
                    match["score"]["opponent"] = opp_score
                    
                    # Complete match button
                    if st.button("Mark as Completed", key=f"complete_{idx}"):
                        match["completed"] = True
                        with open('afcdata.json', 'w', encoding='utf-8') as f:
                            json.dump(st.session_state.data, f, indent=2, ensure_ascii=False)
                        st.rerun()
                
                else:
                    # Display match summary
                    st.subheader("Match Summary")
                    st.write(f"Score: AFC {match['score']['AFC']} - {match['score']['opponent']} {match['opponent']}")
                    
                    # Goals and assists
                    if match["goals"]:
                        st.subheader("Goals")
                        for goal in match["goals"]:
                            assist_text = f" (assist: {goal['assist']})" if goal['assist'] else ""
                            st.write(f"⚽ {goal['scorer']}{assist_text}")
                    
                    # Cards
                    if match["cards"]:
                        st.subheader("Cards")
                        for card in match["cards"]:
                            emoji = "🟨" if card["type"] == "yellow" else "🟥"
                            st.write(f"{emoji} {card['player']}")
                    
                    # Best performers
                    st.subheader("Best Performers")
                    top_ratings = sorted(
                        match["ratings"].items(),
                        key=lambda x: x[1],
                        reverse=True
                    )[:3]
                    for player, rating in top_ratings:
                        st.write(f"{player}: {rating}/10")
                    
                    if match["motm"]:
                        st.write(f"👑 Man of the Match: {match['motm']}")
                
                # Display lineup
                st.subheader("Lineup")
                pitch = draw_football_pitch(match["formation"])
                positions = FORMATIONS[match["formation"]]["positions"]
                coordinates = FORMATIONS[match["formation"]]["coordinates"]
                
                for pos, coord in zip(positions, coordinates):
                    player = match["players"].get(pos, "")
                    if player:
                        text = f"{player} ({match['numbers'][player]})"
                        if player == match["captain"]:
                            text += " ©️"
                        pitch.add_annotation(
                            x=coord[0], y=coord[1],
                            text=text,
                            showarrow=False,
                            font=dict(color="white"),
                            bgcolor="rgba(0,0,139,0.7)"
                        )
                st.plotly_chart(pitch, use_container_width=True)
                
                # Delete match button
                if st.button("Delete Match", key=f"delete_match_{idx}"):
                    st.session_state.data["matches"].pop(idx)
                    with open('afcdata.json', 'w', encoding='utf-8') as f:
                        json.dump(st.session_state.data, f, indent=2, ensure_ascii=False)
                    st.rerun()