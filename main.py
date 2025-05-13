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

# Title and description
st.title("⚽ Premier League Match Predictor")
st.markdown("""
    Predict the outcome of Premier League matches using machine learning models.
    Select the teams and model to get predictions.
""")

# Load the data with optimized caching
@st.cache_data(ttl=3600)  # Cache for 1 hour
def load_data():
    try:
        # Load all necessary columns for feature preparation
        df = pd.read_csv("final_dataset.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        return None

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
    
    # Create three columns for team and model selection
    col1, col2, col3 = st.columns(3)
    
    with col1:
        home_team = st.selectbox("Select Home Team", teams)
    
    with col2:
        away_team = st.selectbox("Select Away Team", teams)
    
    with col3:
        selected_model = st.selectbox("Select Model", list(models.keys()))
    
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
                
                # Prepare features based on the selected model
                if selected_model == 'XGBoost':
                    # Get the exact feature order from the training data
                    training_features = [
                        'Unnamed: 0', 'FTHG', 'FTAG', 'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP',
                        'MW', 'HTFormPts', 'ATFormPts', 'HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3',
                        'HTLossStreak5', 'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5',
                        'HTGD', 'ATGD', 'DiffPts', 'DiffFormPts', 'Year', 'Month', 'Day', 'DayOfWeek'
                    ]
                    
                    # Get the exact team names from the training data
                    home_teams = sorted(df['HomeTeam'].unique())
                    away_teams = sorted(df['AwayTeam'].unique())
                    
                    # Create team features in the exact order
                    team_features = []
                    for team in home_teams:
                        team_features.append(f'HomeTeam_{team}')
                    for team in away_teams:
                        team_features.append(f'AwayTeam_{team}')
                    
                    # Add form features in the exact order
                    form_features = [
                        'HM1_M', 'HM2_M', 'HM3_M', 'HM4_M', 'HM5_M',
                        'AM1_M', 'AM2_M', 'AM3_M', 'AM4_M', 'AM5_M',
                        'HTFormPtsStr_MMMMM', 'ATFormPtsStr_MMMMM'
                    ]
                    
                    # Combine all features in the exact order
                    all_features = training_features + team_features + form_features
                    
                    # Create a new DataFrame with all features initialized to 0
                    final_input = pd.DataFrame(0, index=[0], columns=all_features)
                    
                    # Copy values from sample_data where features exist
                    for col in sample_data.columns:
                        if col in final_input.columns:
                            final_input[col] = sample_data[col]
                    
                    # Set the selected teams to 1
                    final_input[f'HomeTeam_{home_team}'] = 1
                    final_input[f'AwayTeam_{away_team}'] = 1
                    
                    # Set form features to 1
                    for feature in form_features:
                        final_input[feature] = 1
                    
                    # Scale numeric features
                    numeric_cols = [col for col in final_input.columns if col not in team_features + form_features]
                    scaler = StandardScaler()
                    final_input[numeric_cols] = scaler.fit_transform(final_input[numeric_cols])
                    
                    input_data = final_input
                    
                elif selected_model == 'Random Forest':
                    # Random Forest specific preparation
                    scaler = StandardScaler()
                    numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
                    sample_data[numeric_cols] = scaler.fit_transform(sample_data[numeric_cols])
                    input_data = sample_data
                else:  # Logistic Regression
                    # Logistic Regression specific preparation
                    scaler = StandardScaler()
                    numeric_cols = sample_data.select_dtypes(include=[np.number]).columns
                    sample_data[numeric_cols] = scaler.fit_transform(sample_data[numeric_cols])
                    input_data = sample_data
                
                # Make prediction with selected model
                model = models[selected_model]
                pred = model.predict_proba(input_data)[0]
                progress_bar.progress(100)
                
                # Display prediction
                st.subheader(f"Match Prediction using {selected_model}")
                
                # Create prediction box
                st.markdown("""
                    <div class="prediction-box">
                """, unsafe_allow_html=True)
                
                # Display probabilities with confidence levels
                home_win_prob = pred[0]
                draw_prob = pred[1]
                away_win_prob = pred[2]
                
                # Calculate confidence level based on probability difference
                max_prob = max(home_win_prob, draw_prob, away_win_prob)
                confidence = "High" if max_prob > 0.6 else "Medium" if max_prob > 0.45 else "Low"
                
                st.metric("Home Win", f"{home_win_prob:.1%}", delta=None, delta_color="normal")
                st.metric("Draw", f"{draw_prob:.1%}", delta=None, delta_color="normal")
                st.metric("Away Win", f"{away_win_prob:.1%}", delta=None, delta_color="normal")
                st.metric("Confidence Level", confidence)
                
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
                    st.dataframe(historical_matches[['Date', 'HomeTeam', 'AwayTeam', 'FTR']])
                    
                    # Create a visualization of historical results
                    fig, ax = plt.subplots(figsize=(10, 6))
                    sns.countplot(data=historical_matches, x='FTR')
                    plt.title(f"Historical Results: {home_team} vs {away_team}")
                    st.pyplot(fig)
                else:
                    st.info("No historical matches found between these teams.")
                
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")
                st.error("Please ensure all required features are present in the dataset.")
    
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
        - XGBoost Accuracy: 78.50%
        - Random Forest Accuracy: 64.47%
        - Logistic Regression Accuracy: [To be added]
    """)
    
    # Add data source information
    st.sidebar.title("Data Source")
    st.sidebar.markdown("""
        The data comes from historical Premier League matches,
        including team form and performance statistics.
    """)
else:
    st.error("Failed to load the required data or models. Please check if the files are in the correct location.")
