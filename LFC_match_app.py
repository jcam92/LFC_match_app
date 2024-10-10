import streamlit as st
import requests
import pandas as pd

# Set up the page title
st.title("Liverpool FC Match Analysis")

# Function to fetch matches
@st.cache_data
def fetch_matches():
    url = "https://api.football-data.org/v4/competitions/PL/matches"
    headers = {
        "X-Auth-Token": "47ef51d81e46467ea979eee380dd6345"
    }

    # Fetch data
    response = requests.get(url, headers=headers)
    
    # Debugging info to be printed in the terminal, not in the Streamlit app
    print(f"Response Status Code: {response.status_code}")
    print(f"Response Content: {response.text}")
    
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Function to process the matches and filter for Liverpool
@st.cache_data
def load_data():
    data = fetch_matches()
    if data:
        matches = data['matches']
        df = pd.json_normalize(matches)
        
        # Filter for Liverpool FC matches
        df_liverpool = df[(df['homeTeam.name'] == 'Liverpool FC') | (df['awayTeam.name'] == 'Liverpool FC')]
        
        # Debugging information for terminal
        print(f"Liverpool FC Matches Dataframe:\n{df_liverpool.head()}")
        
        return df_liverpool
    else:
        return pd.DataFrame()

# Load data
df_flat = load_data()

if not df_flat.empty:
    # Display the data
    st.write(df_flat[['utcDate', 'homeTeam.name', 'awayTeam.name', 'score.fullTime.homeTeam', 'score.fullTime.awayTeam']])

    # Calculate and display statistics
    st.subheader("Liverpool Match Statistics:")

    wins = df_flat[
        ((df_flat['homeTeam.name'] == 'Liverpool FC') & (df_flat['score.fullTime.homeTeam'] > df_flat['score.fullTime.awayTeam'])) |
        ((df_flat['awayTeam.name'] == 'Liverpool FC') & (df_flat['score.fullTime.awayTeam'] > df_flat['score.fullTime.homeTeam']))
    ].shape[0]
    
    draws = df_flat[df_flat['score.fullTime.homeTeam'] == df_flat['score.fullTime.awayTeam']].shape[0]
    
    losses = df_flat[
        ((df_flat['homeTeam.name'] == 'Liverpool FC') & (df_flat['score.fullTime.homeTeam'] < df_flat['score.fullTime.awayTeam'])) |
        ((df_flat['awayTeam.name'] == 'Liverpool FC') & (df_flat['score.fullTime.awayTeam'] < df_flat['score.fullTime.homeTeam']))
    ].shape[0]

    st.write(f"Wins: {wins}")
    st.write(f"Draws: {draws}")
    st.write(f"Losses: {losses}")

else:
    st.write("No matches found or unable to fetch data.")
