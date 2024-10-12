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


