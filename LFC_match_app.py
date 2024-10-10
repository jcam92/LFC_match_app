import pandas as pd
import streamlit as st
import requests

# Function to fetch data
def fetch_matches():
    url = 'https://api.football-data.org/v4/competitions/PL/matches'
    headers = {'X-Auth-Token': '47ef51d81e46467ea979eee380dd6345'}
    response = requests.get(url, headers=headers)
    data = response.json()
    return data

# Load and flatten data
@st.cache_data
def load_data():
    data = fetch_matches()
    matches = data['matches']
    df = pd.json_normalize(matches)
    return df

df_flat = load_data()

# Debugging step: Show the available columns in df_flat
st.write(df_flat.columns)

# Define the required columns
required_columns = ['utcDate', 'homeTeam.name', 'awayTeam.name', 'score.fullTime.homeTeam', 'score.fullTime.awayTeam']
missing_columns = [col for col in required_columns if col not in df_flat.columns]

# Display missing columns or the table
if missing_columns:
    st.write(f"Missing columns in the dataset: {missing_columns}")
else:
    st.write(df_flat[required_columns])

# Calculate and display stats
wins = len(df_flat[df_flat['score.fullTime.homeTeam'] > df_flat['score.fullTime.awayTeam']])
draws = len(df_flat[df_flat['score.fullTime.homeTeam'] == df_flat['score.fullTime.awayTeam']])
losses = len(df_flat[df_flat['score.fullTime.homeTeam'] < df_flat['score.fullTime.awayTeam']])

st.write("Liverpool Match Statistics:")
st.write(f"Wins: {wins}")
st.write(f"Draws: {draws}")
st.write(f"Losses: {losses}")
