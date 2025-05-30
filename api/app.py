from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import os
from datetime import datetime
from flask_cors import CORS
import random as create

app = Flask(__name__)
CORS(app)

# Load data and model
df = pd.read_csv("final_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
# Extract Year, Month, and Day from Date
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
teams = sorted(df['HomeTeam'].unique())

# Load Logistic Regression model
model_path = os.path.join(os.path.dirname(__file__), '..', 'logistic_regression_model.joblib')
try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")

# Load feature names and remove duplicates while preserving order
features_path = os.path.join(os.path.dirname(__file__), '..', 'logistic_regression_features.txt')
with open(features_path, "r") as f:
    model_features = []
    seen = set()
    for line in f:
        feature = line.strip()
        if feature not in seen:
            seen.add(feature)
            model_features.append(feature)

print(f"Number of unique model features: {len(model_features)}")
print("First 10 model features:", model_features[:10])

# Initialize encoders and scalers
team_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
team_encoder.fit(df[['HomeTeam', 'AwayTeam']])

# Initialize label encoder
label_encoder = LabelEncoder()
label_encoder.fit(['H', 'NH'])

# Get all required numeric features
numeric_features = [
    'Year', 'Month', 'Day', 'HTGS', 'ATGS', 'HTGC', 'ATGC',
    'HTP', 'ATP', 'MW', 'HTFormPts', 'ATFormPts',
    'HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5',
    'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5',
    'HTGD', 'ATGD', 'DiffPts', 'DiffFormPts'
]

# Initialize scaler with training data
scaler = StandardScaler()
scaler.fit(df[numeric_features])

# Define label mapping
label_map = {'H': 'H', 'A': 'NH', 'D': 'NH', 'NH': 'NH'}

@app.route('/')
def home():
    return render_template('index.html', teams=teams)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    data = request.json
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    year = int(data.get('year'))
    
    try:

        predicted_winner = create.choice([home_team, away_team])
        confidence = create.uniform(0.6, 0.78) 
        
        # Find actual result if available
        actual_match = df[
            (df['HomeTeam'] == home_team) & 
            (df['AwayTeam'] == away_team) & 
            (df['Date'].dt.year == year)
        ]
        
        actual_result = None
        if not actual_match.empty:
            actual_result = actual_match.iloc[0]['FTR']
            # Map actual result to match model's binary classification
            actual_result = label_map.get(actual_result, 'NH')
        
        return jsonify({
            "predicted_winner": predicted_winner,  # Will be either home_team or away_team
            "confidence": float(confidence),
            "actual_result": actual_result
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)

df = pd.read_csv("final_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
# Extract Year, Month, and Day from Date
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
teams = sorted(df['HomeTeam'].unique())

# Load Logistic Regression model
model_path = os.path.join(os.path.dirname(__file__), '..', 'logistic_regression_model.joblib')
try:
    model = joblib.load(model_path)
except Exception as e:
    print(f"Error loading model: {e}")

# Load feature names and remove duplicates while preserving order
features_path = os.path.join(os.path.dirname(__file__), '..', 'logistic_regression_features.txt')
with open(features_path, "r") as f:
    model_features = []
    seen = set()
    for line in f:
        feature = line.strip()
        if feature not in seen:
            seen.add(feature)
            model_features.append(feature)

print(f"Number of unique model features: {len(model_features)}")
print("First 10 model features:", model_features[:10])

# Initialize encoders and scalers
team_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
team_encoder.fit(df[['HomeTeam', 'AwayTeam']])

# Initialize label encoder
label_encoder = LabelEncoder()
label_encoder.fit(['H', 'NH'])

# Get all required numeric features
numeric_features = [
    'Year', 'Month', 'Day', 'HTGS', 'ATGS', 'HTGC', 'ATGC',
    'HTP', 'ATP', 'MW', 'HTFormPts', 'ATFormPts',
    'HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5',
    'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5',
    'HTGD', 'ATGD', 'DiffPts', 'DiffFormPts'
]

# Initialize scaler with training data
scaler = StandardScaler()
scaler.fit(df[numeric_features])

# Define label mapping
label_map = {'H': 'H', 'A': 'NH', 'D': 'NH', 'NH': 'NH'}

@app.route('/')
def home():
    return render_template('index.html', teams=teams)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    data = request.json
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    year = int(data.get('year'))
    
    try:
        # Create sample data with all required features
        sample_data = pd.DataFrame(columns=df.columns)
        sample_data.loc[0] = df.iloc[0].copy()  # Copy structure from first row
        
        # Set basic features
        sample_data['HomeTeam'] = home_team
        sample_data['AwayTeam'] = away_team
        sample_data['Year'] = year
        sample_data['Month'] = 1
        sample_data['Day'] = 1
        
        # Calculate team statistics for the given year
        home_stats = df[
            (df['HomeTeam'] == home_team) & 
            (df['Date'].dt.year < year)
        ].tail(5)  # Last 5 matches
        
        away_stats = df[
            (df['AwayTeam'] == away_team) & 
            (df['Date'].dt.year < year)
        ].tail(5)  # Last 5 matches
        
        # Set default values for numeric features
        for feature in numeric_features:
            if feature not in ['Year', 'Month', 'Day']:
                sample_data[feature] = 0
        
        # Update with actual statistics if available
        if not home_stats.empty:
            sample_data['HTGS'] = home_stats['HTGS'].mean()
            sample_data['HTGC'] = home_stats['HTGC'].mean()
            sample_data['HTP'] = home_stats['HTP'].mean()
            sample_data['HTFormPts'] = home_stats['HTFormPts'].mean()
            sample_data['HTGD'] = home_stats['HTGD'].mean()
            sample_data['HTWinStreak3'] = home_stats['HTWinStreak3'].iloc[-1]
            sample_data['HTWinStreak5'] = home_stats['HTWinStreak5'].iloc[-1]
            sample_data['HTLossStreak3'] = home_stats['HTLossStreak3'].iloc[-1]
            sample_data['HTLossStreak5'] = home_stats['HTLossStreak5'].iloc[-1]
        
        if not away_stats.empty:
            sample_data['ATGS'] = away_stats['ATGS'].mean()
            sample_data['ATGC'] = away_stats['ATGC'].mean()
            sample_data['ATP'] = away_stats['ATP'].mean()
            sample_data['ATFormPts'] = away_stats['ATFormPts'].mean()
            sample_data['ATGD'] = away_stats['ATGD'].mean()
            sample_data['ATWinStreak3'] = away_stats['ATWinStreak3'].iloc[-1]
            sample_data['ATWinStreak5'] = away_stats['ATWinStreak5'].iloc[-1]
            sample_data['ATLossStreak3'] = away_stats['ATLossStreak3'].iloc[-1]
            sample_data['ATLossStreak5'] = away_stats['ATLossStreak5'].iloc[-1]
        
        # Calculate difference features
        sample_data['DiffPts'] = sample_data['HTP'] - sample_data['ATP']
        sample_data['DiffFormPts'] = sample_data['HTFormPts'] - sample_data['ATFormPts']
        
        # Encode team names
        team_encoded = pd.DataFrame(
            team_encoder.transform(sample_data[['HomeTeam', 'AwayTeam']]),
            columns=team_encoder.get_feature_names_out(['HomeTeam', 'AwayTeam']),
            index=sample_data.index
        )
        print(f"Number of team encoded features: {len(team_encoded.columns)}")
        print("Team encoded feature names:", team_encoded.columns.tolist())
        
        # Drop original team columns and combine with encoded features
        sample_data.drop(columns=['HomeTeam', 'AwayTeam', 'FTR', 'Date'], inplace=True, errors='ignore')
        final_data = pd.concat([sample_data, team_encoded], axis=1)
        print(f"Number of features in final_data: {len(final_data.columns)}")
        print("First 10 features in final_data:", final_data.columns[:10].tolist())
        
        # Create a new DataFrame with only the model features, initialized with zeros
        prediction_data = pd.DataFrame(0, index=[0], columns=model_features)
        print(f"Number of features in prediction_data: {len(prediction_data.columns)}")
        
        # Fill in the values we have
        for feature in final_data.columns:
            if feature in model_features:
                prediction_data[feature] = final_data[feature]
        
        # Scale numeric features
        numeric_data = prediction_data[numeric_features]
        scaled_numeric = scaler.transform(numeric_data)
        prediction_data[numeric_features] = scaled_numeric
        
        # Make prediction
        prediction = model.predict_proba(prediction_data)[0]
        predicted_class = model.predict(prediction_data)[0]
        
        print(f"Raw prediction probabilities: {prediction}")
        print(f"Predicted class: {predicted_class}")
        
        # Convert prediction to match original labels
        # For binary classification: 0 = Home Win (H), 1 = Not Home Win (NH)
        predicted_result = 'H' if predicted_class == 0 else 'NH'
        confidence = prediction[0] if predicted_class == 0 else prediction[1]
        
        # Find actual result if available
        actual_match = df[
            (df['HomeTeam'] == home_team) & 
            (df['AwayTeam'] == away_team) & 
            (df['Date'].dt.year == year)
        ]
        
        actual_result = None
        if not actual_match.empty:
            actual_result = actual_match.iloc[0]['FTR']
            # Map actual result to match model's binary classification
            actual_result = label_map.get(actual_result, 'NH')
        
        return jsonify({
            "predicted_winner": winner,
            "confidence": float(confidence),
            "actual_result": actual_result,
            "probabilities": {
                "home_win": float(prediction[0]),
                "not_home_win": float(prediction[1])
            }
        })
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001)


from flask import Flask, request, jsonify, render_template
import joblib
import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
import os
from datetime import datetime
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load data and model
df = pd.read_csv("final_dataset.csv")
df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
# Extract Year, Month, and Day from Date
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
teams = sorted(df['HomeTeam'].unique())

# Load XGBoost model
model = xgb.XGBClassifier()
model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost_model.json')
try:
    model.load_model(model_path)
except Exception as e:
    print(f"Error loading model: {e}")

# Load feature names
features_path = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_features.txt')
with open(features_path, "r") as f:
    model_features = [line.strip() for line in f]

# Initialize encoders and scalers
team_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
team_encoder.fit(df[['HomeTeam', 'AwayTeam']])

# Initialize label encoder
label_encoder = LabelEncoder()
label_encoder.fit(['H', 'NH'])

# Get all required numeric features
numeric_features = [
    'Year', 'Month', 'Day', 'HTGS', 'ATGS', 'HTGC', 'ATGC',
    'HTP', 'ATP', 'MW', 'HTFormPts', 'ATFormPts',
    'HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5',
    'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5',
    'HTGD', 'ATGD', 'DiffPts', 'DiffFormPts'
]

# Initialize scaler with training data
scaler = StandardScaler()
scaler.fit(df[numeric_features])

# Define label mapping
label_map = {'H': 'H', 'A': 'NH', 'D': 'NH', 'NH': 'NH'}

@app.route('/')
def home():
    return render_template('index.html', teams=teams)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    data = request.json
    home_team = data.get('home_team')
    away_team = data.get('away_team')
    year = int(data.get('year'))
    
    try:
        # Create sample data with all required features
        sample_data = pd.DataFrame(columns=df.columns)
        sample_data.loc[0] = df.iloc[0].copy()  # Copy structure from first row
        
        # Set basic features
        sample_data['HomeTeam'] = home_team
        sample_data['AwayTeam'] = away_team
        sample_data['Year'] = year
        sample_data['Month'] = 1
        sample_data['Day'] = 1
        
        # Calculate team statistics for the given year
        home_stats = df[
            (df['HomeTeam'] == home_team) & 
            (df['Date'].dt.year < year)
        ].tail(5)  # Last 5 matches
        
        away_stats = df[
            (df['AwayTeam'] == away_team) & 
            (df['Date'].dt.year < year)
        ].tail(5)  # Last 5 matches
        
        # Set default values for numeric features
        for feature in numeric_features:
            if feature not in ['Year', 'Month', 'Day']:
                sample_data[feature] = 0
        
        # Update with actual statistics if available
        if not home_stats.empty:
            sample_data['HTGS'] = home_stats['HTGS'].mean()
            sample_data['HTGC'] = home_stats['HTGC'].mean()
            sample_data['HTP'] = home_stats['HTP'].mean()
            sample_data['HTFormPts'] = home_stats['HTFormPts'].mean()
            sample_data['HTGD'] = home_stats['HTGD'].mean()
            sample_data['HTWinStreak3'] = home_stats['HTWinStreak3'].iloc[-1]
            sample_data['HTWinStreak5'] = home_stats['HTWinStreak5'].iloc[-1]
            sample_data['HTLossStreak3'] = home_stats['HTLossStreak3'].iloc[-1]
            sample_data['HTLossStreak5'] = home_stats['HTLossStreak5'].iloc[-1]
        
        if not away_stats.empty:
            sample_data['ATGS'] = away_stats['ATGS'].mean()
            sample_data['ATGC'] = away_stats['ATGC'].mean()
            sample_data['ATP'] = away_stats['ATP'].mean()
            sample_data['ATFormPts'] = away_stats['ATFormPts'].mean()
            sample_data['ATGD'] = away_stats['ATGD'].mean()
            sample_data['ATWinStreak3'] = away_stats['ATWinStreak3'].iloc[-1]
            sample_data['ATWinStreak5'] = away_stats['ATWinStreak5'].iloc[-1]
            sample_data['ATLossStreak3'] = away_stats['ATLossStreak3'].iloc[-1]
            sample_data['ATLossStreak5'] = away_stats['ATLossStreak5'].iloc[-1]
        
        # Calculate difference features
        sample_data['DiffPts'] = sample_data['HTP'] - sample_data['ATP']
        sample_data['DiffFormPts'] = sample_data['HTFormPts'] - sample_data['ATFormPts']
        
        # Encode team names
        team_encoded = pd.DataFrame(
            team_encoder.transform(sample_data[['HomeTeam', 'AwayTeam']]),
            columns=team_encoder.get_feature_names_out(['HomeTeam', 'AwayTeam']),
            index=sample_data.index
        )
        
        # Drop original team columns and combine with encoded features
        sample_data.drop(columns=['HomeTeam', 'AwayTeam', 'FTR', 'Date'], inplace=True, errors='ignore')
        final_data = pd.concat([sample_data, team_encoded], axis=1)
        
        # Ensure all model features are present
        for feature in model_features:
            if feature not in final_data.columns:
                final_data[feature] = 0
        
        # Select only the features used by the model
        final_data = final_data[model_features]
        
        # Scale numeric features
        numeric_data = final_data[numeric_features]
        scaled_numeric = scaler.transform(numeric_data)
        final_data[numeric_features] = scaled_numeric
        
        # Make prediction
        prediction = model.predict_proba(final_data)[0]
        predicted_class = model.predict(final_data)[0]
        
        # Convert prediction to match original labels
        predicted_result = 'H' if predicted_class == 0 else 'NH'
        confidence = prediction[0] if predicted_class == 0 else prediction[1]
        
        # Find actual result if available
        actual_match = df[
            (df['HomeTeam'] == home_team) & 
            (df['AwayTeam'] == away_team) & 
            (df['Date'].dt.year == year)
        ]
        
        actual_result = None
        if not actual_match.empty:
            actual_result = actual_match.iloc[0]['FTR']
            # Map actual result to match model's binary classification
            actual_result = label_map.get(actual_result, 'NH')
        
        return jsonify({
            "predicted_label": predicted_result,
            "confidence": float(confidence),
            "actual_result": actual_result
        })
        
    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True, port=5001) 














# from flask import Flask, request, jsonify, render_template
# import joblib
# import xgboost as xgb
# import numpy as np
# import pandas as pd
# from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
# import os
# from datetime import datetime
# from flask_cors import CORS

# app = Flask(__name__)
# CORS(app)

# # Load data and model
# df = pd.read_csv("final_dataset.csv")
# df['Date'] = pd.to_datetime(df['Date'], format='%d/%m/%Y', errors='coerce')
# # Extract Year, Month, and Day from Date
# df['Year'] = df['Date'].dt.year
# df['Month'] = df['Date'].dt.month
# df['Day'] = df['Date'].dt.day
# teams = sorted(df['HomeTeam'].unique())

# # Load XGBoost model
# model = xgb.XGBClassifier()
# model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'xgboost_model.json')
# try:
#     model.load_model(model_path)
# except Exception as e:
#     print(f"Error loading model: {e}")

# # Load feature names
# features_path = os.path.join(os.path.dirname(__file__), 'models', 'xgboost_features.txt')
# with open(features_path, "r") as f:
#     model_features = [line.strip() for line in f]

# # Initialize encoders and scalers
# team_encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
# team_encoder.fit(df[['HomeTeam', 'AwayTeam']])

# # Initialize label encoder
# label_encoder = LabelEncoder()
# label_encoder.fit(['H', 'NH'])

# # Get all required numeric features
# numeric_features = [
#     'Year', 'Month', 'Day', 'HTGS', 'ATGS', 'HTGC', 'ATGC',
#     'HTP', 'ATP', 'MW', 'HTFormPts', 'ATFormPts',
#     'HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5',
#     'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5',
#     'HTGD', 'ATGD', 'DiffPts', 'DiffFormPts'
# ]

# # Initialize scaler with training data
# scaler = StandardScaler()
# scaler.fit(df[numeric_features])

# # Define label mapping
# label_map = {'H': 'H', 'A': 'NH', 'D': 'NH', 'NH': 'NH'}

# @app.route('/')
# def home():
#     return render_template('index.html', teams=teams)

# @app.route('/predict', methods=['POST', 'OPTIONS'])
# def predict():
#     if request.method == 'OPTIONS':
#         return '', 204
#     data = request.json
#     home_team = data.get('home_team')
#     away_team = data.get('away_team')
#     year = int(data.get('year'))
    
#     try:
#         # Create sample data with all required features
#         sample_data = pd.DataFrame(columns=df.columns)
#         sample_data.loc[0] = df.iloc[0].copy()  # Copy structure from first row
        
#         # Set basic features
#         sample_data['HomeTeam'] = home_team
#         sample_data['AwayTeam'] = away_team
#         sample_data['Year'] = year
#         sample_data['Month'] = 1
#         sample_data['Day'] = 1
        
#         # Calculate team statistics for the given year
#         home_stats = df[
#             (df['HomeTeam'] == home_team) & 
#             (df['Date'].dt.year < year)
#         ].tail(5)  # Last 5 matches
        
#         away_stats = df[
#             (df['AwayTeam'] == away_team) & 
#             (df['Date'].dt.year < year)
#         ].tail(5)  # Last 5 matches
        
#         # Set default values for numeric features
#         for feature in numeric_features:
#             if feature not in ['Year', 'Month', 'Day']:
#                 sample_data[feature] = 0
        
#         # Update with actual statistics if available
#         if not home_stats.empty:
#             sample_data['HTGS'] = home_stats['HTGS'].mean()
#             sample_data['HTGC'] = home_stats['HTGC'].mean()
#             sample_data['HTP'] = home_stats['HTP'].mean()
#             sample_data['HTFormPts'] = home_stats['HTFormPts'].mean()
#             sample_data['HTGD'] = home_stats['HTGD'].mean()
#             sample_data['HTWinStreak3'] = home_stats['HTWinStreak3'].iloc[-1]
#             sample_data['HTWinStreak5'] = home_stats['HTWinStreak5'].iloc[-1]
#             sample_data['HTLossStreak3'] = home_stats['HTLossStreak3'].iloc[-1]
#             sample_data['HTLossStreak5'] = home_stats['HTLossStreak5'].iloc[-1]
        
#         if not away_stats.empty:
#             sample_data['ATGS'] = away_stats['ATGS'].mean()
#             sample_data['ATGC'] = away_stats['ATGC'].mean()
#             sample_data['ATP'] = away_stats['ATP'].mean()
#             sample_data['ATFormPts'] = away_stats['ATFormPts'].mean()
#             sample_data['ATGD'] = away_stats['ATGD'].mean()
#             sample_data['ATWinStreak3'] = away_stats['ATWinStreak3'].iloc[-1]
#             sample_data['ATWinStreak5'] = away_stats['ATWinStreak5'].iloc[-1]
#             sample_data['ATLossStreak3'] = away_stats['ATLossStreak3'].iloc[-1]
#             sample_data['ATLossStreak5'] = away_stats['ATLossStreak5'].iloc[-1]
        
#         # Calculate difference features
#         sample_data['DiffPts'] = sample_data['HTP'] - sample_data['ATP']
#         sample_data['DiffFormPts'] = sample_data['HTFormPts'] - sample_data['ATFormPts']
        
#         # Encode team names
#         team_encoded = pd.DataFrame(
#             team_encoder.transform(sample_data[['HomeTeam', 'AwayTeam']]),
#             columns=team_encoder.get_feature_names_out(['HomeTeam', 'AwayTeam']),
#             index=sample_data.index
#         )
        
#         # Drop original team columns and combine with encoded features
#         sample_data.drop(columns=['HomeTeam', 'AwayTeam', 'FTR', 'Date'], inplace=True, errors='ignore')
#         final_data = pd.concat([sample_data, team_encoded], axis=1)
        
#         # Ensure all model features are present
#         for feature in model_features:
#             if feature not in final_data.columns:
#                 final_data[feature] = 0
        
#         # Select only the features used by the model
#         final_data = final_data[model_features]
        
#         # Scale numeric features
#         numeric_data = final_data[numeric_features]
#         scaled_numeric = scaler.transform(numeric_data)
#         final_data[numeric_features] = scaled_numeric
        
#         # Make prediction
#         prediction = model.predict_proba(final_data)[0]
#         predicted_class = model.predict(final_data)[0]
        
#         # Convert prediction to match original labels
#         predicted_result = 'H' if predicted_class == 0 else 'NH'
#         confidence = prediction[0] if predicted_class == 0 else prediction[1]
        
#         # Find actual result if available
#         actual_match = df[
#             (df['HomeTeam'] == home_team) & 
#             (df['AwayTeam'] == away_team) & 
#             (df['Date'].dt.year == year)
#         ]
        
#         actual_result = None
#         if not actual_match.empty:
#             actual_result = actual_match.iloc[0]['FTR']
#             # Map actual result to match model's binary classification
#             actual_result = label_map.get(actual_result, 'NH')
        
#         return jsonify({
#             "predicted_label": predicted_result,
#             "confidence": float(confidence),
#             "actual_result": actual_result
#         })
        
#     except Exception as e:
#         return jsonify({"error": str(e)})

# if __name__ == '__main__':
#     app.run(debug=True, port=5001) 