import streamlit as st
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import numpy as np

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
        
        # Add new features
        df_liverpool['is_home'] = df_liverpool['homeTeam.name'] == 'Liverpool FC'
        df_liverpool['opponent'] = df_liverpool.apply(lambda row: row['awayTeam.name'] if row['is_home'] else row['homeTeam.name'], axis=1)
        df_liverpool['goal_difference'] = df_liverpool.apply(lambda row: row['score.fullTime.home'] - row['score.fullTime.away'] if row['is_home'] else row['score.fullTime.away'] - row['score.fullTime.home'], axis=1)
        df_liverpool['result'] = df_liverpool['goal_difference'].apply(lambda x: 'win' if x > 0 else ('draw' if x == 0 else 'loss'))
         # Add form (last 3 matches)
        df_liverpool['form'] = df_liverpool['result'].rolling(window=3, min_periods=1).apply(lambda x: sum(x == 'win') - sum(x == 'loss')).shift(1)
    
    # Add goal scoring and conceding averages
        df_liverpool['avg_goals_scored'] = df_liverpool['goal_difference'].apply(lambda x: max(x, 0)).rolling(window=3, min_periods=1).mean().shift(1)
        df_liverpool['avg_goals_conceded'] = df_liverpool['goal_difference'].apply(lambda x: max(-x, 0)).rolling(window=3, min_periods=1).mean().shift(1)
        return df_liverpool
    else:
        return pd.DataFrame()

# Load data
df_flat = load_data()

if not df_flat.empty:
    # Display the data using correct column names
    st.write(df_flat[['utcDate', 'homeTeam.name', 'awayTeam.name', 'score.fullTime.home', 'score.fullTime.away']])

    # Calculate and display statistics
    st.subheader("Liverpool Match Statistics:")

    wins = df_flat[
        ((df_flat['homeTeam.name'] == 'Liverpool FC') & (df_flat['score.fullTime.home'] > df_flat['score.fullTime.away'])) |
        ((df_flat['awayTeam.name'] == 'Liverpool FC') & (df_flat['score.fullTime.away'] > df_flat['score.fullTime.home']))
    ].shape[0]
    
    draws = df_flat[df_flat['score.fullTime.home'] == df_flat['score.fullTime.away']].shape[0]
    
    losses = df_flat[
        ((df_flat['homeTeam.name'] == 'Liverpool FC') & (df_flat['score.fullTime.home'] < df_flat['score.fullTime.away'])) |
        ((df_flat['awayTeam.name'] == 'Liverpool FC') & (df_flat['score.fullTime.away'] < df_flat['score.fullTime.home']))
    ].shape[0]

    st.write(f"Wins: {wins}")
    st.write(f"Draws: {draws}")
    st.write(f"Losses: {losses}")

# Prepare data for machine learning
features = ['is_home', 'form', 'avg_goals_scored', 'avg_goals_conceded']
X = pd.get_dummies(df_flat[features + ['opponent']], columns=['opponent'])
y = df_flat['result']

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train a Random Forest Classifier
    rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

    # After splitting the data
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

    # Train the model on the resampled data
    rf_classifier.fit(X_train_resampled, y_train_resampled)

    # Make predictions
    y_pred = rf_classifier.predict(X_test_scaled)
    # Display model performance
    st.subheader("Machine Learning Model Performance:")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.2f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
    st.write("Classification Report:")
    st.code(classification_report(y_test, y_pred))

    # Allow user to make predictions
    st.subheader("Predict Next Match:")
    opponent = st.selectbox("Select opponent:", df_flat['opponent'].unique())
    is_home = st.checkbox("Is it a home game?")

    if st.button("Predict"):
        try:
            input_data = pd.DataFrame({'is_home': [is_home], 'form': [0], 'avg_goals_scored': [0], 'avg_goals_conceded': [0]})
            input_data = pd.get_dummies(input_data.assign(opponent=opponent), columns=['opponent'])
            input_data = input_data.reindex(columns=X.columns, fill_value=0)
            input_scaled = scaler.transform(input_data)
            prediction = rf_classifier.predict(input_scaled)
            probabilities = rf_classifier.predict_proba(input_scaled)[0]
            st.write(f"Predicted outcome: {prediction[0]}")
            st.write(f"Probabilities: Win: {probabilities[2]:.2f}, Draw: {probabilities[0]:.2f}, Loss: {probabilities[1]:.2f}")
        except Exception as e:
            st.write(f"An error occurred: {str(e)}")

    # After training the model
    feature_importance = rf_classifier.feature_importances_
    feature_names = X.columns

    # Sort features by importance
    sorted_idx = feature_importance.argsort()
    pos = np.arange(sorted_idx.shape[0]) + .5

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(pos, feature_importance[sorted_idx], align='center')
    ax.set_yticks(pos)
    ax.set_yticklabels(feature_names[sorted_idx])
    ax.set_xlabel('Feature Importance')
    ax.set_title('Feature Importance for Liverpool FC Match Prediction')

    st.pyplot(fig)

else:
    st.write("No matches found or unable to fetch data.")

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
svm_classifier = SVC(probability=True, random_state=42)

ensemble_classifier = VotingClassifier(
    estimators=[('rf', rf_classifier), ('gb', gb_classifier), ('svm', svm_classifier)],
    voting='soft'
)

# Train the ensemble model
ensemble_classifier.fit(X_train_resampled, y_train_resampled)

# Make predictions using the ensemble
y_pred = ensemble_classifier.predict(X_test_scaled)

