import streamlit as st
import requests
import pandas as pd

# Set up API token and base URL
API_TOKEN = '47ef51d81e46467ea979eee380dd6345'
BASE_URL = 'https://api.football-data.org/v4/'

# Fetch matches with debugging output
@st.cache_data
def fetch_matches():
    headers = {'X-Auth-Token': API_TOKEN}
    url = BASE_URL + 'competitions/PL/matches'
    
    response = requests.get(url, headers=headers)

    # Debugging output
    st.write(f"Fetching data from {url}")
    st.write(f"Response Status Code: {response.status_code}")
    st.write(f"Response Content: {response.text}")  # Show the raw response

    if response.status_code != 200:
        st.error(f"Error fetching data: {response.status_code} {response.text}")
        return []

    data = response.json()
    return data.get('matches', [])

# Load data into a dataframe
def load_data():
    matches = fetch_matches()
    if matches:
        df = pd.json_normalize(matches, sep='_')
        df = df[['utcDate', 'homeTeam_name', 'awayTeam_name', 'score_fullTime_home', 'score_fullTime_away']]  # Corrected typo here
        df['match_result'] = df.apply(
            lambda x: 1 if x['score_fullTime_home'] > x['score_fullTime_away'] 
            else (-1 if x['score_fullTime_home'] < x['score_fullTime_away'] 
                  else 0), axis=1)
        return df
    return pd.DataFrame()

# Streamlit app layout
st.title('Liverpool FC Match Analysis')

# Check for data
df_flat = load_data()
if not df_flat.empty:
    # Filter LFC matches
    liverpool_matches = df_flat[(df_flat['homeTeam_name'] == 'Liverpool FC') | (df_flat['awayTeam_name'] == 'Liverpool FC')].copy()
    liverpool_matches['liverpool_home'] = liverpool_matches['homeTeam_name'] == 'Liverpool FC'

    # Calculate match outcomes (1 = win, -1 = loss, 0 = draw)
    liverpool_matches['liverpool_result'] = liverpool_matches.apply(
        lambda x: 1 if (x['liverpool_home'] and x['match_result'] == 1) or
                      (not x['liverpool_home'] and x['match_result'] == -1)
        else (-1 if (x['liverpool_home'] and x['match_result'] == -1) or
                   (not x['liverpool_home'] and x['match_result'] == 1)
              else 0), axis=1)

    # Display the filtered data
    st.dataframe(liverpool_matches[['utcDate', 'homeTeam_name', 'awayTeam_name', 'score_fullTime_home', 'score_fullTime_away', 'liverpool_result']])

    # Stat summary
    st.write("Liverpool Match Statistics:")
    wins = liverpool_matches[liverpool_matches['liverpool_result'] == 1].shape[0]
    draws = liverpool_matches[liverpool_matches['liverpool_result'] == 0].shape[0]
    losses = liverpool_matches[liverpool_matches['liverpool_result'] == -1].shape[0]

    st.write(f"Wins: {wins}")
    st.write(f"Draws: {draws}")
    st.write(f"Losses: {losses}")

else:
    st.write("No matches found or unable to fetch data.")


