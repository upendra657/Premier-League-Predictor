import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Set page configuration
st.set_page_config(
    page_title="Premier League Match Predictor",
    page_icon="⚽",
    layout="wide"
)

# Load XGBoost model
@st.cache_resource(ttl=3600)
def load_model():
    try:
        xgb_model = xgb.XGBClassifier()
        xgb_model.load_model("xgboost_model.json")
        return xgb_model
    except Exception as e:
        st.error(f"Error loading XGBoost model: {str(e)}")
        return None

# Load feature names for XGBoost
@st.cache_data(ttl=3600)
def load_feature_names():
    try:
        with open("xgboost_features.txt", "r") as f:
            return [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        st.error("Feature list file not found. Please ensure 'xgboost_features.txt' is in the project directory.")
        return []

# Load dataset
@st.cache_data(ttl=3600)
def load_data():
    try:
        df = pd.read_csv("final_dataset.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

# App title
st.title("⚽ Premier League Match Predictor - XGBoost")
st.markdown("Predict outcomes using an XGBoost model trained on Premier League data.")

# Load data, model, and feature names
df = load_data()
model = load_model()
feature_names = load_feature_names()

if df is not None and model is not None and feature_names:
    teams = sorted(df['HomeTeam'].unique())

    col1, col2 = st.columns(2)
    with col1:
        home_team = st.selectbox("Select Home Team", teams, key="home_team_unique")
    with col2:
        away_team = st.selectbox("Select Away Team", teams, key="away_team_unique")

    if st.button("Predict Match Outcome"):
        if home_team == away_team:
            st.error("Please select different teams for home and away.")
        else:
            try:
                sample_data = df.iloc[0:1].copy()
                sample_data['HomeTeam'] = home_team
                sample_data['AwayTeam'] = away_team

                current_date = datetime.now()
                sample_data['Year'] = current_date.year
                sample_data['Month'] = current_date.month
                sample_data['Day'] = current_date.day
                sample_data.drop(columns='Date', inplace=True, errors='ignore')

                # Drop target if exists
                sample_data.drop(columns='FTR', inplace=True, errors='ignore')

                categorical_cols = ['HomeTeam', 'AwayTeam']
                encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                encoded = pd.DataFrame(
                    encoder.fit_transform(sample_data[categorical_cols]),
                    columns=encoder.get_feature_names_out(categorical_cols),
                    index=sample_data.index
                )

                sample_data.drop(columns=categorical_cols, inplace=True)
                final_data = pd.concat([sample_data, encoded], axis=1)

                # Align with model features
                for col in feature_names:
                    if col not in final_data.columns:
                        final_data[col] = 0
                final_data = final_data[feature_names]

                scaler = StandardScaler()
                final_data_scaled = scaler.fit_transform(final_data)

                prediction = model.predict_proba(final_data_scaled)[0]
                outcome_labels = ['H', 'D', 'A']
                predicted_index = np.argmax(prediction)
                predicted_result = outcome_labels[predicted_index]
                confidence = prediction[predicted_index]

                result_text = home_team if predicted_result == 'H' else away_team if predicted_result == 'A' else 'Draw'

                st.subheader("Prediction Result")
                st.metric("Predicted Outcome", result_text)
                st.metric("Prediction Confidence", f"{confidence:.2%}")

                # Add historical match information
                st.subheader("Historical Match Information")

                # Filter historical matches between these teams (cached)
                @st.cache_data
                def get_historical_matches(home, away):
                    return df[
                        ((df['HomeTeam'] == home) & (df['AwayTeam'] == away)) |
                        ((df['HomeTeam'] == away) & (df['AwayTeam'] == home))
                    ]

                historical_matches = get_historical_matches(home_team, away_team)

                if not historical_matches.empty:
                    # Display historical matches
                    st.write("### Recent Head-to-Head Matches")
                    st.dataframe(historical_matches[['Date', 'HomeTeam', 'AwayTeam', 'FTR']].sort_values('Date', ascending=False))

                    # Create a visualization of historical results
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(data=historical_matches, x='FTR')
                    plt.title(f"Historical Results: {home_team} vs {away_team}")
                    st.pyplot(fig)

                    # Add historical statistics
                    st.write("### Historical Statistics")
                    total_matches = len(historical_matches)
                    home_wins = len(historical_matches[historical_matches['FTR'] == 'H'])
                    draws = len(historical_matches[historical_matches['FTR'] == 'D'])
                    away_wins = len(historical_matches[historical_matches['FTR'] == 'A'])

                    st.write(f"Total matches played: {total_matches}")
                    st.write(f"Home wins: {home_wins} ({home_wins/total_matches:.1%})")
                    st.write(f"Draws: {draws} ({draws/total_matches:.1%})")
                    st.write(f"Away wins: {away_wins} ({away_wins/total_matches:.1%})")
                else:
                    st.info("No historical matches found between these teams.")

            except Exception as e:
                st.error(f"Prediction error: {str(e)}")
else:
    st.error("Failed to load model or data.")

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #f0f2f6;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)


# Load models with optimized caching
@st.cache_resource(ttl=3600)  # Cache for 1 hour
def load_models():
    try:
        models = {}
        model_files = {
            'XGBoost': 'xgboost_model.json',
            'Random Forest': 'random_forest_model.joblib',
            'Logistic Regression': 'logistic_regression_model.joblib'
        }
        
        # Load XGBoost model
        if os.path.exists(model_files['XGBoost']):
            try:
                xgb_model = xgb.XGBClassifier()
                xgb_model.load_model(model_files['XGBoost'])
                models['XGBoost'] = xgb_model
            except Exception as e:
                st.warning(f"Could not load XGBoost model: {str(e)}")
        
        # Load Random Forest model
        if os.path.exists(model_files['Random Forest']):
            try:
                rf_model = joblib.load(model_files['Random Forest'])
                models['Random Forest'] = rf_model
            except Exception as e:
                st.warning(f"Could not load Random Forest model: {str(e)}")
        
        # Load Logistic Regression model
        if os.path.exists(model_files['Logistic Regression']):
            try:
                lr_model = joblib.load(model_files['Logistic Regression'])
                models['Logistic Regression'] = lr_model
            except Exception as e:
                st.warning(f"Could not load Logistic Regression model: {str(e)}")
        
        if not models:
            raise Exception("No models could be loaded successfully")
            
        return models
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return {}

# Load data and models
df = load_data()
models = load_models()

if df is not None and models:
    # Get unique teams (cached)
    @st.cache_data
    def get_teams():
        return sorted(df['HomeTeam'].unique())
    
    teams = get_teams()
    
    # # Create three columns for team and model selection
    # col1, col2, col3 = st.columns(3)
    
    # with col1:
    #     home_team = st.selectbox("Select Home Team", teams)
    
    # with col2:
    #     away_team = st.selectbox("Select Away Team", teams)
    
    # with col3:
    #     selected_model = st.selectbox("Select Model", ["Logistic Regression", "XGBoost"], key="model_selection")
    
    # Add a predict button
    if st.button("Predict Match Outcome"):
        if home_team == away_team:
            st.error("Please select different teams for home and away.")
        else:
            # Create a progress bar
            progress_bar = st.progress(0)
            
            try:
                # Create a sample row with the same structure as training data
                sample_data = df.iloc[0:1].copy()
                
                # Update the team names
                sample_data['HomeTeam'] = home_team
                sample_data['AwayTeam'] = away_team
                
                # Set date to current date and convert to numeric features
                current_date = datetime.now()
                sample_data['Year'] = current_date.year
                sample_data['Month'] = current_date.month
                sample_data['Day'] = current_date.day
                sample_data['DayOfWeek'] = current_date.weekday()
                
                # Drop the Date column if it exists
                if 'Date' in sample_data.columns:
                    sample_data = sample_data.drop('Date', axis=1)
                
                # Initialize all required numeric features to 0
                required_numeric_features = [
                    'FTHG', 'FTAG', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP', 'MW',
                    'HTFormPts', 'ATFormPts', 'HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3',
                    'HTLossStreak5', 'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3',
                    'ATLossStreak5', 'HTGD', 'ATGD', 'DiffPts', 'DiffFormPts'
                ]
                
                for feature in required_numeric_features:
                    if feature not in sample_data.columns:
                        sample_data[feature] = 0
                
                # Initialize all required categorical features
                required_categorical_features = {
                    'HM1': 'M', 'HM2': 'M', 'HM3': 'M', 'HM4': 'M', 'HM5': 'M',
                    'AM1': 'M', 'AM2': 'M', 'AM3': 'M', 'AM4': 'M', 'AM5': 'M',
                    'HTFormPtsStr': 'MMMMM', 'ATFormPtsStr': 'MMMMM'
                }
                
                for feature, default_value in required_categorical_features.items():
                    if feature not in sample_data.columns:
                        sample_data[feature] = default_value
                
                # Drop the target variable if it exists
                if 'FTR' in sample_data.columns:
                    sample_data = sample_data.drop('FTR', axis=1)
                
                # One-hot encode categorical variables
                categorical_cols = ['HomeTeam', 'AwayTeam', 'HM1', 'HM2', 'HM3', 'HM4', 'HM5',
                                  'AM1', 'AM2', 'AM3', 'AM4', 'AM5', 'HTFormPtsStr', 'ATFormPtsStr']
                
                # Create a new DataFrame for encoded features
                encoded_data = pd.DataFrame()
                
                # First, encode the team names
                team_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                team_cols = ['HomeTeam', 'AwayTeam']
                team_encoded = pd.DataFrame(
                    team_encoder.fit_transform(sample_data[team_cols]),
                    columns=team_encoder.get_feature_names_out(team_cols),
                    index=sample_data.index
                )
                encoded_data = pd.concat([encoded_data, team_encoded], axis=1)
                
                # Then encode the form features
                form_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
                form_cols = ['HM1', 'HM2', 'HM3', 'HM4', 'HM5', 'AM1', 'AM2', 'AM3', 'AM4', 'AM5',
                           'HTFormPtsStr', 'ATFormPtsStr']
                form_encoded = pd.DataFrame(
                    form_encoder.fit_transform(sample_data[form_cols]),
                    columns=form_encoder.get_feature_names_out(form_cols),
                    index=sample_data.index
                )
                encoded_data = pd.concat([encoded_data, form_encoded], axis=1)
                
                # Add numeric features
                numeric_cols = [col for col in sample_data.columns if col not in categorical_cols]
                numeric_data = sample_data[numeric_cols].copy()
                
                # Scale numeric features
                scaler = StandardScaler()
                numeric_data = pd.DataFrame(
                    scaler.fit_transform(numeric_data),
                    columns=numeric_data.columns,
                    index=numeric_data.index
                )
                
                # Combine all features
                final_data = pd.concat([numeric_data, encoded_data], axis=1)
                
                # Load model features for XGBoost to ensure alignment
                model_features = load_feature_names()
                
                # Add missing features with default 0
                for feature in model_features:
                    if feature not in final_data.columns:
                        final_data[feature] = 0

                # Drop extra columns not seen during training
                extra_features = [col for col in final_data.columns if col not in model_features]
                final_data.drop(columns=extra_features, inplace=True)

                # Reorder to match model feature order
                final_data = final_data[model_features]
                
                input_data = final_data
                
                # Make prediction with selected model
                model = models[selected_model]
                
                # Add debug information
                st.write("Debug Information:")
                st.write(f"Input data shape: {input_data.shape}")
                st.write(f"Model type: {type(model).__name__}")
                st.write("Input data columns:", input_data.columns.tolist())
                st.write("Model features:", model_features)
                
                pred = model.predict_proba(input_data)[0]
                # Insert: extract predicted outcome
                predicted_index = np.argmax(pred)
                outcome_labels = ['H', 'D', 'A']
                predicted_label = outcome_labels[predicted_index]
                predicted_team = home_team if predicted_label == 'H' else away_team if predicted_label == 'A' else "Draw"
                progress_bar.progress(100)
                
                # Display prediction
                st.subheader(f"Match Prediction using {selected_model}")
                
                # Create prediction box
                st.markdown("""
                    <div class="prediction-box">
                """, unsafe_allow_html=True)
                
                home_win_prob = pred[0]
                draw_prob = pred[1]
                away_win_prob = pred[2]
                max_prob = max(home_win_prob, draw_prob, away_win_prob)
                
                # Simplified prediction results
                st.write(f"### Prediction Results for {home_team} vs {away_team}")
                st.metric("Predicted Outcome", predicted_team)
                st.metric("Prediction Confidence", f"{max_prob:.1%}")
                st.metric("Model Accuracy", "78.16%")
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # Add historical match information
                st.subheader("Historical Match Information")
                
                # Filter historical matches between these teams (cached)
                @st.cache_data
                def get_historical_matches(home, away):
                    return df[
                        ((df['HomeTeam'] == home) & (df['AwayTeam'] == away)) |
                        ((df['HomeTeam'] == away) & (df['AwayTeam'] == home))
                    ]
                
                historical_matches = get_historical_matches(home_team, away_team)
                
                if not historical_matches.empty:
                    # Display historical matches
                    st.write("### Recent Head-to-Head Matches")
                    st.dataframe(historical_matches[['Date', 'HomeTeam', 'AwayTeam', 'FTR']].sort_values('Date', ascending=False))
                    
                    # Create a visualization of historical results
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(data=historical_matches, x='FTR')
                    plt.title(f"Historical Results: {home_team} vs {away_team}")
                    st.pyplot(fig)
                    
                    # Add historical statistics
                    st.write("### Historical Statistics")
                    total_matches = len(historical_matches)
                    home_wins = len(historical_matches[historical_matches['FTR'] == 'H'])
                    draws = len(historical_matches[historical_matches['FTR'] == 'D'])
                    away_wins = len(historical_matches[historical_matches['FTR'] == 'A'])
                    
                    st.write(f"Total matches played: {total_matches}")
                    st.write(f"Home wins: {home_wins} ({home_wins/total_matches:.1%})")
                    st.write(f"Draws: {draws} ({draws/total_matches:.1%})")
                    st.write(f"Away wins: {away_wins} ({away_wins/total_matches:.1%})")
                else:
                    st.info("No historical matches found between these teams.")
                    
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.error("Please check the debug information above for more details.")
    
    # Add information about the models
    st.sidebar.title("About the Models")
    st.sidebar.markdown("""
        This application uses machine learning models to predict match outcomes:
        
        - **XGBoost**: A gradient boosting framework
        - **Random Forest**: An ensemble learning method
        - **Logistic Regression**: A linear model for classification
    """)
    
    # Add model performance metrics
    st.sidebar.title("Model Performance")
    st.sidebar.markdown("""
        - XGBoost Accuracy: 78.16%
        - Random Forest Accuracy: 64.47%
        - Logistic Regression Accuracy: 65.86%
    """)
    
    # Add data source information
    st.sidebar.title("Data Source")
    st.sidebar.markdown("""
        The data comes from historical Premier League matches,
        including team form and performance statistics.
    """)
else:
    st.error("Failed to load the required data or models. Please check if the files are in the correct location.")



# import streamlit as st
# import pandas as pd
# import numpy as np
# import joblib
# import os
# from datetime import datetime
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.preprocessing import OneHotEncoder, StandardScaler
# from sklearn.linear_model import LogisticRegression

# # Set page configuration
# st.set_page_config(
#     page_title="Premier League Match Predictor",
#     page_icon="⚽",
#     layout="wide"
# )

# # Load the logistic regression model
# @st.cache_resource(ttl=3600)
# def load_model():
#     try:
#         return joblib.load("logistic_regression_model.joblib")
#     except Exception as e:
#         st.error(f"Error loading Logistic Regression model: {str(e)}")
#         return None

# # Load dataset
# @st.cache_data(ttl=3600)
# def load_data():
#     try:
#         df = pd.read_csv("final_dataset.csv")
#         df['Date'] = pd.to_datetime(df['Date'])
#         return df
#     except Exception as e:
#         st.error(f"Error loading data: {str(e)}")
#         return None

# # App title
# st.title("⚽ Premier League Match Predictor - Logistic Regression")
# st.markdown("Predict outcomes using a logistic regression model trained on Premier League data.")

# # Load data and model
# df = load_data()
# model = load_model()

# if df is not None and model is not None:
#     teams = sorted(df['HomeTeam'].unique())

#     col1, col2 = st.columns(2)
#     with col1:
#         home_team = st.selectbox("Select Home Team", teams)
#     with col2:
#         away_team = st.selectbox("Select Away Team", teams)

#     if st.button("Predict Match Outcome"):
#         if home_team == away_team:
#             st.error("Please select different teams for home and away.")
#         else:
#             try:
#                 sample_data = df.iloc[0:1].copy()
#                 sample_data['HomeTeam'] = home_team
#                 sample_data['AwayTeam'] = away_team

#                 current_date = datetime.now()
#                 sample_data['Year'] = current_date.year
#                 sample_data['Month'] = current_date.month
#                 sample_data['Day'] = current_date.day
#                 sample_data.drop(columns='Date', inplace=True, errors='ignore')

#                 # Drop target if exists
#                 sample_data.drop(columns='FTR', inplace=True, errors='ignore')

#                 categorical_cols = ['HomeTeam', 'AwayTeam']
#                 encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
#                 encoded = pd.DataFrame(
#                     encoder.fit_transform(sample_data[categorical_cols]),
#                     columns=encoder.get_feature_names_out(categorical_cols),
#                     index=sample_data.index
#                 )

#                 sample_data.drop(columns=categorical_cols, inplace=True)
#                 final_data = pd.concat([sample_data, encoded], axis=1)

#                 # Align with model features
#                 model_features = list(model.feature_names_in_)
#                 for col in model_features:
#                     if col not in final_data.columns:
#                         final_data[col] = 0
#                 final_data = final_data[model_features]

#                 scaler = StandardScaler()
#                 final_data_scaled = scaler.fit_transform(final_data)

#                 prediction = model.predict_proba(final_data_scaled)[0]
#                 outcome_labels = ['H', 'D', 'A']
#                 predicted_index = np.argmax(prediction)
#                 predicted_result = outcome_labels[predicted_index]
#                 confidence = prediction[predicted_index]

#                 result_text = home_team if predicted_result == 'H' else away_team if predicted_result == 'A' else 'Draw'

#                 st.subheader("Prediction Result")
#                 st.metric("Predicted Outcome", result_text)
#                 st.metric("Prediction Confidence", f"{confidence:.2%}")

#             except Exception as e:
#                 st.error(f"Prediction error: {str(e)}")
# else:
#     st.error("Failed to load model or data.")

