import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
import os

# === Config ===
INPUT_FILES = ['cleaned_2020_2021.csv', 'cleaned_2021_2022.csv']
OUTPUT_FILE = 'final_testing_dataset_xgb_new.csv'
FEATURE_LIST_FILE = 'xgboost_features.txt'

# === Step 1: Load and Combine Datasets ===
dfs = []
for file in INPUT_FILES:
    if os.path.exists(file):
        df = pd.read_csv(file)
        dfs.append(df)
    else:
        raise FileNotFoundError(f"❌ File not found: {file}")

combined_df = pd.concat(dfs, ignore_index=True)
print(f"✅ Loaded and combined datasets. Total rows: {len(combined_df)}")

# === Step 2: Process Date Columns ===
if 'Date' in combined_df.columns:
    combined_df['Date'] = pd.to_datetime(combined_df['Date'], errors='coerce')
    combined_df['Year'] = combined_df['Date'].dt.year
    combined_df['Month'] = combined_df['Date'].dt.month
    combined_df['Day'] = combined_df['Date'].dt.day
    combined_df.drop(columns='Date', inplace=True)

# === Step 3: Drop Target & Leakage Columns ===
drop_cols = ['FTHG', 'FTAG', 'FTR',
             'HM1', 'HM2', 'HM3', 'HM4', 'HM5',
             'AM1', 'AM2', 'AM3', 'AM4', 'AM5',
             'HTFormPtsStr', 'ATFormPtsStr']

combined_df.drop(columns=[c for c in drop_cols if c in combined_df.columns], inplace=True)

# === Step 4: Fill missing numeric columns ===
required_numerics = [
    'HTGS', 'ATGS', 'HTGC', 'ATGC', 'HTP', 'ATP',
    'MW', 'HTFormPts', 'ATFormPts',
    'HTWinStreak3', 'HTWinStreak5', 'HTLossStreak3', 'HTLossStreak5',
    'ATWinStreak3', 'ATWinStreak5', 'ATLossStreak3', 'ATLossStreak5',
    'HTGD', 'ATGD', 'DiffPts', 'DiffFormPts'
]

for col in required_numerics:
    if col not in combined_df.columns:
        combined_df[col] = 0

# === Step 5: One-hot Encode HomeTeam & AwayTeam ===
team_cols = ['HomeTeam', 'AwayTeam']
if not all(col in combined_df.columns for col in team_cols):
    raise ValueError("❌ Required columns 'HomeTeam' and 'AwayTeam' not found in the dataset.")

# Load team encoder from training features
if os.path.exists(FEATURE_LIST_FILE):
    with open(FEATURE_LIST_FILE) as f:
        expected_features = [line.strip() for line in f.readlines()]
    team_features = [col for col in expected_features if col.startswith('HomeTeam_') or col.startswith('AwayTeam_')]
    known_teams = sorted(list(set(col.split('_', 1)[1] for col in team_features)))
    dummy_df = pd.DataFrame({'HomeTeam': known_teams, 'AwayTeam': known_teams})
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoder.fit(dummy_df[team_cols])
else:
    raise FileNotFoundError("❌ 'xgboost_features.txt' not found. Run training script first.")

encoded_df = pd.DataFrame(
    encoder.transform(combined_df[team_cols]),
    columns=encoder.get_feature_names_out(team_cols),
    index=combined_df.index
)

combined_df.drop(columns=team_cols, inplace=True)
combined_df = pd.concat([combined_df, encoded_df], axis=1)

# === Step 6: Align to Training Feature Set ===
for col in expected_features:
    if col not in combined_df.columns:
        combined_df[col] = 0

combined_df = combined_df[expected_features]

# === Step 7: Save Final Test Dataset ===
combined_df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Final test dataset saved to '{OUTPUT_FILE}' with shape {combined_df.shape}")