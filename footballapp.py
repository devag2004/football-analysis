import streamlit as st
from statsbombpy import sb
import pandas as pd
import numpy as np
from mplsoccer.pitch import Pitch
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Set up Streamlit app layout
st.title("Football Match Analysis and Shot Outcome Prediction")

# Section 1: Competition and Match Selection
st.sidebar.header("Select Competition and Match")
competitions = sb.competitions()
competition_name = st.sidebar.selectbox("Select Competition", competitions['competition_name'].unique())
season_id = st.sidebar.selectbox("Select Season", competitions[competitions['competition_name'] == competition_name]['season_id'].unique())

matches = sb.matches(competition_id=int(competitions[competitions['competition_name'] == competition_name]['competition_id'].iloc[0]), season_id=int(season_id))
match_name = st.sidebar.selectbox("Select Match", matches['match_id'].apply(lambda x: f"{matches[matches['match_id'] == x]['home_team'].values[0]} vs {matches[matches['match_id'] == x]['away_team'].values[0]}").unique())
match_id = matches[matches.apply(lambda row: f"{row['home_team']} vs {row['away_team']}", axis=1) == match_name]['match_id'].values[0]

# Load match events
events = sb.events(match_id=match_id)

# Section 2: Match Summary
st.subheader("Match Summary")
match_info = matches[matches['match_id'] == match_id].iloc[0]
home_team = match_info['home_team']
away_team = match_info['away_team']
venue = match_info['stadium']
date = match_info['match_date']

# Calculate match statistics for both teams
def get_team_stats(team_name):
    team_events = events[events['team'] == team_name]
    goals = team_events[(team_events['type'] == 'Shot') & (team_events['shot_outcome'] == 'Goal')].shape[0]
    
    # Calculate possession based on passes
    total_passes = team_events[team_events['type'] == 'Pass'].shape[0]
    
    # Calculate additional stats
    shots_on_target = team_events[(team_events['type'] == 'Shot') & ((team_events['shot_outcome'] == 'Goal') | (team_events['shot_outcome'] == 'Saved'))].shape[0]
    shots_missed = team_events[(team_events['type'] == 'Shot') & (team_events['shot_outcome'] == 'Off T')].shape[0]
    yellow_cards = team_events[team_events['type'] == 'Yellow Card'].shape[0]
    red_cards = team_events[team_events['type'] == 'Red Card'].shape[0]
    
    return goals, total_passes, shots_on_target, shots_missed, yellow_cards, red_cards

home_goals, home_passes, home_shots_on_target, home_shots_missed, home_yellow_cards, home_red_cards = get_team_stats(home_team)
away_goals, away_passes, away_shots_on_target, away_shots_missed, away_yellow_cards, away_red_cards = get_team_stats(away_team)

# Calculate possession percentage
total_passes = home_passes + away_passes
home_possession = (home_passes / total_passes * 100) if total_passes > 0 else 0
away_possession = (away_passes / total_passes * 100) if total_passes > 0 else 0

# Display match summary side-by-side for both teams
st.write(f"**Date:** {date}")
st.write(f"**Stadium:** {venue}")
col1, col2 = st.columns(2)

with col1:
    st.write(f"### {home_team}")
    st.write(f"**Goals:** {home_goals}")
    st.write(f"**Possession:** {home_possession:.2f}%")
    st.write(f"**Shots on Target:** {home_shots_on_target}")
    st.write(f"**Shots Off Target:** {home_shots_missed}")
    st.write(f"**Yellow Cards:** {home_yellow_cards}")
    st.write(f"**Red Cards:** {home_red_cards}")
    st.write(f"**Passes:** {home_passes}")

with col2:
    st.write(f"### {away_team}")
    st.write(f"**Goals:** {away_goals}")
    st.write(f"**Possession:** {away_possession:.2f}%")
    st.write(f"**Shots on Target:** {away_shots_on_target}")
    st.write(f"**Shots Off Target:** {away_shots_missed}")
    st.write(f"**Yellow Cards:** {away_yellow_cards}")
    st.write(f"**Red Cards:** {away_red_cards}")
    st.write(f"**Passes:** {away_passes}")

# Section 3: Player-Specific Metrics
st.subheader("Player-Specific Metrics")
players = events['player'].unique()
selected_player = st.selectbox("Select Player", players)

# Get player-specific stats
def get_player_stats(player_name):
    player_events = events[events['player'] == player_name]
    goals = player_events[(player_events['type'] == 'Shot') & (player_events['shot_outcome'] == 'Goal')].shape[0]
    assists = player_events[player_events['type'] == 'Pass'].shape[0]  # Assuming all passes are assists for simplicity
    total_passes = player_events[player_events['type'] == 'Pass'].shape[0]
    shots_on_target = player_events[(player_events['type'] == 'Shot') & ((player_events['shot_outcome'] == 'Goal') | (player_events['shot_outcome'] == 'Saved'))].shape[0]
    shots_missed = player_events[(player_events['type'] == 'Shot') & (player_events['shot_outcome'] == 'Off T')].shape[0]

    return goals, assists, total_passes, shots_on_target, shots_missed

player_goals, player_assists, player_passes, player_shots_on_target, player_shots_missed = get_player_stats(selected_player)

# Display player-specific stats
st.write(f"### Stats for {selected_player}")
st.write(f"**Goals:** {player_goals}")
st.write(f"**Total Passes:** {player_passes}")
st.write(f"**Shots on Target:** {player_shots_on_target}")
st.write(f"**Shots Off Target:** {player_shots_missed}")

# Section 4: Data Visualizations
st.header("Team Data:")
selected_team = st.selectbox("Select Team for Visualization", [home_team, away_team])
st.subheader("Team Formation Visualization")
team_players = events[(events['team'] == selected_team) & (events['type'] == 'Lineup')]

# Ensure there's lineup data to work with
if not team_players.empty:
    # Extract player names and positions from the lineup event
    player_positions = team_players['player'].tolist()
    positions = team_players['position'].tolist()

    # Define a mapping of formations (you can adjust as necessary)
    formation_positions = {
        'Goalkeeper': (10, 50),
        'Defender': [(30, 30), (30, 50), (30, 70)],  # Adjust as needed
        'Midfielder': [(50, 30), (50, 50), (50, 70)],  # Adjust as needed
        'Forward': [(70, 30), (70, 50)],  # Adjust as needed
    }

    # Create pitch for formation
    pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
    fig, ax = pitch.draw(figsize=(10, 6))

    # Plot players based on their positions
    for player, pos in zip(player_positions, positions):
        if pos in formation_positions:
            # If multiple positions, iterate through them
            if isinstance(formation_positions[pos], list):
                for i, (x, y) in enumerate(formation_positions[pos]):
                    ax.text(x, y + i * 5, player, ha='center', va='center', fontsize=12, color='black',
                            bbox=dict(facecolor='white', alpha=0.5))
            else:
                x, y = formation_positions[pos]
                ax.text(x, y, player, ha='center', va='center', fontsize=12, color='black',
                        bbox=dict(facecolor='white', alpha=0.5))

    # Show formation plot
    st.pyplot(fig)
else:
    st.write("No lineup data available for this team.")

# Heatmap

st.subheader("Heatmap")
team_events = events[events['team'] == selected_team]

pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
fig, ax = pitch.draw(figsize=(10, 6))
pitch.kdeplot(team_events['location'].dropna().apply(lambda x: x[0]), team_events['location'].dropna().apply(lambda x: x[1]), ax=ax, fill=True, cmap='coolwarm', levels=100)
st.pyplot(fig)

# Pass Network
st.subheader("Pass Network")
passes = team_events[team_events['type'] == 'Pass']

# Create separate DataFrames for successful and unsuccessful passes
successful_passes = passes[passes['pass_outcome'].isnull()]  # Successful passes have no 'pass_outcome'
unsuccessful_passes = passes[passes['pass_outcome'].notnull()]  # Unsuccessful passes have a 'pass_outcome'

# Create the pitch for the pass network
pitch = Pitch(pitch_type='statsbomb', pitch_color='grass', line_color='white')
fig, ax = pitch.draw(figsize=(10, 6))

# Plot successful passes in blue
pitch.arrows(successful_passes['location'].apply(lambda x: x[0]),
             successful_passes['location'].apply(lambda x: x[1]),
             successful_passes['pass_end_location'].apply(lambda x: x[0]),
             successful_passes['pass_end_location'].apply(lambda x: x[1]),
             width=2, headwidth=4, color="blue", ax=ax)

# Plot unsuccessful passes in red
pitch.arrows(unsuccessful_passes['location'].apply(lambda x: x[0]),
             unsuccessful_passes['location'].apply(lambda x: x[1]),
             unsuccessful_passes['pass_end_location'].apply(lambda x: x[0]),
             unsuccessful_passes['pass_end_location'].apply(lambda x: x[1]),
             width=2, headwidth=4, color="red", ax=ax)

# Show the plot
handles = [
    plt.Line2D([0], [0], color='blue', lw=4, label='Successful Passes'),
    plt.Line2D([0], [0], color='red', lw=4, label='Unsuccessful Passes')
]
ax.legend(handles=handles, loc='upper right')
st.pyplot(fig)

# Goals & Misses Visualization on Goal
st.subheader("Goals and Misses on Goalpost")
shots = events[events['type'] == 'Shot']

# Create the goal column based on shot outcome
shots['goal'] = shots['shot_outcome'].apply(lambda x: 1 if x == 'Goal' else 0)

goal_shots = shots[shots['shot_end_location'].notna()]
goal_shots['end_x'] = goal_shots['shot_end_location'].apply(lambda x: x[0])
goal_shots['end_y'] = goal_shots['shot_end_location'].apply(lambda x: x[1])

fig, ax = plt.subplots(figsize=(8, 4))
pitch = Pitch(goal_type='box', pitch_color='grass', line_color='white', stripe=True, goal_alpha=0.2)
pitch.draw(ax)
sc = ax.scatter(goal_shots['end_x'], goal_shots['end_y'], c=goal_shots['goal'], cmap='coolwarm', s=100, edgecolors='black')
plt.colorbar(sc, ax=ax, label='Goal (1) / Miss (0)')
st.pyplot(fig)

# Section 5: Predict Shot Outcome
st.subheader("Shot Outcome Prediction")

# Convert timestamp to datetime if it's in string format
if shots['timestamp'].dtype == 'O':  # Check if the dtype is object (string)
    shots['timestamp'] = pd.to_datetime(shots['timestamp'], errors='coerce')

# Feature engineering for shot prediction model
# Calculate shot distance and angle
shots['shot_distance'] = np.sqrt((shots['location'].apply(lambda x: x[0]) - 120)**2 + 
                                   (shots['location'].apply(lambda x: x[1]) - 40)**2)

shots['shot_angle'] = np.arctan2(shots['location'].apply(lambda x: x[1]) - 40, 
                                   shots['location'].apply(lambda x: x[0]) - 120) * (180 / np.pi)

# Additional features
shots['is_on_target'] = shots['shot_outcome'].apply(lambda x: 1 if x in ['Goal', 'Saved'] else 0)  # Indicator if shot was on target
shots['time_to_shot'] = shots['timestamp'].apply(lambda x: x.minute + x.second / 60 if pd.notnull(x) else 0)  # Time in minutes

# Prepare feature matrix and target vector
X = shots[['shot_distance', 'shot_angle', 'is_on_target', 'time_to_shot']]
y = shots['goal'].astype(int)

# Split the data into training and test sets for better validation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a Random Forest Classifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# User input for shot distance, angle, and additional features
shot_distance = st.number_input("Enter Shot Distance (in meters)", min_value=0.0, max_value=50.0, value=20.0)
shot_angle = st.number_input("Enter Shot Angle (in degrees)", min_value=0.0, max_value=90.0, value=45.0)

# Assuming the shot is taken at the current minute of the match for the sake of example
current_time_minute = st.number_input("Enter Current Match Minute", min_value=0, max_value=90, value=45)

# Scale user input
user_input = np.array([[shot_distance, shot_angle, 1, current_time_minute]])  # 1 for is_on_target
user_input_scaled = scaler.transform(user_input)

# Predict outcome based on user input
predicted_probability = model.predict_proba(user_input_scaled)[0][1]

st.write(f"**Probability of Scoring:** {predicted_probability:.2%}")

# Provide additional feedback based on shot distance and probability
if predicted_probability >= 0.75:
    st.write("High likelihood of scoring based on match data!")
elif predicted_probability >= 0.5:
    st.write("Moderate chance of scoring based on match data.")
else:
    st.write("Low probability of scoring based on match data.")

# Additional context on shot distance
if shot_distance < 10:
    st.write("Shots taken from this distance have a high probability of scoring!")
elif shot_distance < 20:
    st.write("This is a moderate distance; probability varies.")
else:
    st.write("Shots from this distance are less likely to score.")
