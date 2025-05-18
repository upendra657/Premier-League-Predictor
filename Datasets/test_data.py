import pandas as pd
import numpy as np
from datetime import datetime

# === Step 1: Load the cleaned season files ===
df_2020 = pd.read_csv("cleaned_2020_2021.csv")
df_2021 = pd.read_csv("cleaned_2021_2022.csv")

# Merge and sort chronologically
df = pd.concat([df_2020, df_2021], ignore_index=True)
df['Date'] = pd.to_datetime(df['Date'], dayfirst=True, errors='coerce')
df = df.sort_values('Date').reset_index(drop=True)

# Drop rows with missing critical values
df = df.dropna(subset=['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG', 'FTR'])

# Feature: Matchweek counter
df['MW'] = df.groupby('Season').cumcount() + 1 if 'Season' in df.columns else np.arange(1, len(df)+1)

# Extract date-based features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.weekday

# Initialize team stat tracking
teams = np.unique(df[['HomeTeam', 'AwayTeam']].values)
team_stats = {
    team: {
        'Points': 0, 'Form': [], 'GoalsScored': 0, 'GoalsConceded': 0,
        'MatchesPlayed': 0
    } for team in teams
}

# Feature containers
ht_pts, at_pts = [], []
ht_gs, at_gs = [], []
ht_gc, at_gc = [], []
ht_form, at_form = [], []
ht_gd, at_gd = [], []
diff_pts, diff_form_pts = [], []
ht_form_str, at_form_str = [], []
hm_series, am_series = [], []

# Generate match-by-match stats
for _, row in df.iterrows():
    home, away = row['HomeTeam'], row['AwayTeam']
    fthg, ftag = row['FTHG'], row['FTAG']

    home_stats = team_stats[home]
    away_stats = team_stats[away]

    # Historical features
    ht_pts.append(home_stats['Points'])
    at_pts.append(away_stats['Points'])
    ht_gs.append(home_stats['GoalsScored'])
    at_gs.append(away_stats['GoalsScored'])
    ht_gc.append(home_stats['GoalsConceded'])
    at_gc.append(away_stats['GoalsConceded'])

    home_form_last5 = home_stats['Form'][-5:]
    away_form_last5 = away_stats['Form'][-5:]

    ht_form.append(sum(home_form_last5))
    at_form.append(sum(away_form_last5))
    ht_form_str.append(''.join(['W' if x == 3 else 'D' if x == 1 else 'L' for x in home_form_last5]).rjust(5, 'M'))
    at_form_str.append(''.join(['W' if x == 3 else 'D' if x == 1 else 'L' for x in away_form_last5]).rjust(5, 'M'))

    ht_gd.append(home_stats['GoalsScored'] - home_stats['GoalsConceded'])
    at_gd.append(away_stats['GoalsScored'] - away_stats['GoalsConceded'])

    diff_pts.append(home_stats['Points'] - away_stats['Points'])
    diff_form_pts.append(sum(home_form_last5) - sum(away_form_last5))

    hm_series.append(['W' if x == 3 else 'D' if x == 1 else 'L' if x == 0 else 'M' for x in home_form_last5][-5:])
    am_series.append(['W' if x == 3 else 'D' if x == 1 else 'L' if x == 0 else 'M' for x in away_form_last5][-5:])

    # Outcome encoding
    if fthg > ftag:
        home_result, away_result = 3, 0
    elif fthg == ftag:
        home_result, away_result = 1, 1
    else:
        home_result, away_result = 0, 3

    # Update stats
    home_stats['Points'] += home_result
    away_stats['Points'] += away_result
    home_stats['GoalsScored'] += fthg
    home_stats['GoalsConceded'] += ftag
    away_stats['GoalsScored'] += ftag
    away_stats['GoalsConceded'] += fthg
    home_stats['Form'].append(home_result)
    away_stats['Form'].append(away_result)
    home_stats['MatchesPlayed'] += 1
    away_stats['MatchesPlayed'] += 1

# Append generated features
df['HTP'] = ht_pts
df['ATP'] = at_pts
df['HTGS'] = ht_gs
df['ATGS'] = at_gs
df['HTGC'] = ht_gc
df['ATGC'] = at_gc
df['HTFormPts'] = ht_form
df['ATFormPts'] = at_form
df['HTFormPtsStr'] = ht_form_str
df['ATFormPtsStr'] = at_form_str
df['HTGD'] = ht_gd
df['ATGD'] = at_gd
df['DiffPts'] = diff_pts
df['DiffFormPts'] = diff_form_pts

for i in range(5):
    df[f'HM{i+1}'] = [x[i] if len(x) > i else 'M' for x in hm_series]
    df[f'AM{i+1}'] = [x[i] if len(x) > i else 'M' for x in am_series]

# Match final_dataset.csv structure
reference = pd.read_csv("final_dataset.csv", nrows=1)
expected_cols = reference.columns.tolist()

for col in expected_cols:
    if col not in df.columns:
        df[col] = 0  # Default value

df = df[[col for col in expected_cols if col in df.columns]]

# Save to disk
df.to_csv("fiinal_testing_dataset.csv", index=False)
print("✅ Saved: fiinal_testing_dataset.csv")