import streamlit as st
import requests
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, classification_report, balanced_accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import numpy as np
import traceback
import seaborn as sns

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
        st.error(f"Failed to fetch data. Status code: {response.status_code}")
        return None

# Function to process the matches and filter for Liverpool
@st.cache_data
def load_data():
    try:
        data = fetch_matches()
        if data:
            matches = data['matches']
            df = pd.json_normalize(matches)
            
            st.write("Data fetched successfully. Shape:", df.shape)
            st.write("Columns:", df.columns)
            
            # Filter for Liverpool FC matches
            df_liverpool = df[(df['homeTeam.name'] == 'Liverpool FC') | (df['awayTeam.name'] == 'Liverpool FC')]
            st.write("Liverpool matches filtered. Shape:", df_liverpool.shape)
            
            # Add new features
            df_liverpool['is_home'] = df_liverpool['homeTeam.name'] == 'Liverpool FC'
            df_liverpool['opponent'] = df_liverpool.apply(lambda row: row['awayTeam.name'] if row['is_home'] else row['homeTeam.name'], axis=1)
            
            # Ensure score columns are numeric and handle unplayed matches
            df_liverpool['score.fullTime.home'] = pd.to_numeric(df_liverpool['score.fullTime.home'], errors='coerce')
            df_liverpool['score.fullTime.away'] = pd.to_numeric(df_liverpool['score.fullTime.away'], errors='coerce')
            
            # Identify unplayed matches
            df_liverpool['is_played'] = df_liverpool['score.fullTime.home'].notna() & df_liverpool['score.fullTime.away'].notna()
            
            st.write("Score columns converted to numeric. Sample data:")
            st.write(df_liverpool[['score.fullTime.home', 'score.fullTime.away', 'is_played']].head())
            
            # Calculate goal difference and result only for played matches
            df_liverpool['goal_difference'] = df_liverpool.apply(lambda row: row['score.fullTime.home'] - row['score.fullTime.away'] if row['is_home'] else row['score.fullTime.away'] - row['score.fullTime.home'], axis=1)
            df_liverpool['result'] = df_liverpool.apply(lambda row: 
                'win' if row['is_played'] and row['goal_difference'] > 0 else
                'draw' if row['is_played'] and row['goal_difference'] == 0 else
                'loss' if row['is_played'] and row['goal_difference'] < 0 else
                'not played', axis=1)
            
            st.write("Goal difference and result calculated. Sample data:")
            st.write(df_liverpool[['goal_difference', 'result', 'is_played']].head())
            
            # Add form (last 3 played matches)
            df_liverpool['form'] = df_liverpool[df_liverpool['is_played']]['result'].replace({'win': 1, 'draw': 0, 'loss': -1}).rolling(window=3, min_periods=1).sum().shift(1)
        
            # Add goal scoring and conceding averages (only for played matches)
            df_liverpool['avg_goals_scored'] = df_liverpool[df_liverpool['is_played']]['goal_difference'].apply(lambda x: max(x, 0)).rolling(window=3, min_periods=1).mean().shift(1)
            df_liverpool['avg_goals_conceded'] = df_liverpool[df_liverpool['is_played']]['goal_difference'].apply(lambda x: max(-x, 0)).rolling(window=3, min_periods=1).mean().shift(1)
            
            st.write("Form and average goals calculated. Sample data:")
            st.write(df_liverpool[['form', 'avg_goals_scored', 'avg_goals_conceded', 'is_played']].head())
            
            # Fill NaN values
            df_liverpool = df_liverpool.fillna(0)
            
            # Remove unplayed matches for model training
            df_liverpool_played = df_liverpool[df_liverpool['is_played']]
            
            st.write("Final dataframe shape (played matches only):", df_liverpool_played.shape)
            st.write("Final dataframe columns:", df_liverpool_played.columns)
            
            return df_liverpool_played
        else:
            st.error("No data returned from fetch_matches()")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"An error occurred in load_data(): {str(e)}")
        st.write("Error traceback:", traceback.format_exc())
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

    # Use LabelEncoder instead of one-hot encoding for the target variable
    le = LabelEncoder()
    y = le.fit_transform(df_flat['result'])

    # Print unique values in the target variable
    st.write("Unique values in the target variable:", le.classes_)
    st.write("Shape of X:", X.shape)
    st.write("Shape of y:", y.shape)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Define classifiers
    rf_classifier = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
    gb_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
    svm_classifier = SVC(probability=True, class_weight='balanced', random_state=42)

    ensemble_classifier = VotingClassifier(
        estimators=[('rf', rf_classifier), ('gb', gb_classifier), ('svm', svm_classifier)],
        voting='soft'
    )

    # Train the ensemble model
    ensemble_classifier.fit(X_train_scaled, y_train)

    # Make predictions using the ensemble
    y_pred = ensemble_classifier.predict(X_test_scaled)

    # Display model performance
    st.subheader("Machine Learning Model Performance:")
    st.write(f"Accuracy: {accuracy_score(y_test, y_pred):.2f}")
    st.write(f"Balanced Accuracy: {balanced_accuracy_score(y_test, y_pred):.2f}")
    st.write(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.2f}")
    st.write("Classification Report:")
    st.code(classification_report(y_test, y_pred, target_names=le.classes_))

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
            prediction = ensemble_classifier.predict(input_scaled)
            probabilities = ensemble_classifier.predict_proba(input_scaled)[0]
            
            st.write(f"Predicted outcome: {le.inverse_transform(prediction)[0]}")
            
            for class_name, prob in zip(le.classes_, probabilities):
                st.write(f"Probability of {class_name}: {prob:.2f}")
            
        except Exception as e:
            st.write(f"An error occurred during prediction: {str(e)}")
            st.write("Error traceback:", traceback.format_exc())

    # Feature importance plot (using Random Forest classifier)
    rf_classifier.fit(X_train_scaled, y_train)
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

    st.subheader("Goal Difference Distribution")
    fig, ax = plt.subplots()
    sns.histplot(df_flat['goal_difference'], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Form vs Result")
    fig, ax = plt.subplots()
    sns.boxplot(x='result', y='form', data=df_flat, ax=ax)
    st.pyplot(fig)

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test.argmax(axis=1), y_pred.argmax(axis=1))
    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt='d', ax=ax)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)

    st.subheader("Team Comparison")
    team1 = st.selectbox("Select first team:", df_flat['opponent'].unique())
    team2 = st.selectbox("Select second team:", df_flat['opponent'].unique())

    team1_data = df_flat[df_flat['opponent'] == team1]
    team2_data = df_flat[df_flat['opponent'] == team2]

    st.write(f"{team1} average goals scored: {team1_data['avg_goals_scored'].mean():.2f}")
    st.write(f"{team2} average goals scored: {team2_data['avg_goals_scored'].mean():.2f}")

else:
    st.write("No matches found or unable to fetch data.")
