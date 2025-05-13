import pandas as pd
import numpy as np
from datetime import datetime

# Read the cleaned datasets
df1 = pd.read_csv('cleaned_2020_2021.csv')
df2 = pd.read_csv('cleaned_2021_2022.csv')

# Combine the datasets
combined_df = pd.concat([df1, df2], ignore_index=True)

# Convert Date to datetime
combined_df['Date'] = pd.to_datetime(combined_df['Date'])

# Sort by date
combined_df = combined_df.sort_values('Date')

# Initialize new columns for historical features
teams = combined_df['HomeTeam'].unique()
for team in teams:
    # Goals scored
    combined_df[f'{team}_GS'] = 0
    # Goals conceded
    combined_df[f'{team}_GC'] = 0
    # Points
    combined_df[f'{team}_Pts'] = 0
    # Form points string
    combined_df[f'{team}_FormPtsStr'] = ''
    # Form points
    combined_df[f'{team}_FormPts'] = 0
    # Win streak
    combined_df[f'{team}_WinStreak3'] = 0
    combined_df[f'{team}_WinStreak5'] = 0
    # Loss streak
    combined_df[f'{team}_LossStreak3'] = 0
    combined_df[f'{team}_LossStreak5'] = 0

# Calculate historical features
for idx, row in combined_df.iterrows():
    date = row['Date']
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    
    # Get previous matches for both teams
    home_prev_matches = combined_df[(combined_df['Date'] < date) & 
                                  ((combined_df['HomeTeam'] == home_team) | 
                                   (combined_df['AwayTeam'] == home_team))]
    away_prev_matches = combined_df[(combined_df['Date'] < date) & 
                                  ((combined_df['HomeTeam'] == away_team) | 
                                   (combined_df['AwayTeam'] == away_team))]
    
    # Calculate home team features
    if not home_prev_matches.empty:
        home_goals_scored = 0
        home_goals_conceded = 0
        home_points = 0
        home_form = []
        
        for _, match in home_prev_matches.iterrows():
            if match['HomeTeam'] == home_team:
                home_goals_scored += match['FTHG']
                home_goals_conceded += match['FTAG']
                if match['FTR'] == 'H':
                    home_points += 3
                    home_form.append('W')
                elif match['FTR'] == 'D':
                    home_points += 1
                    home_form.append('D')
                else:
                    home_form.append('L')
            else:
                home_goals_scored += match['FTAG']
                home_goals_conceded += match['FTHG']
                if match['FTR'] == 'A':
                    home_points += 3
                    home_form.append('W')
                elif match['FTR'] == 'D':
                    home_points += 1
                    home_form.append('D')
                else:
                    home_form.append('L')
        
        combined_df.at[idx, f'{home_team}_GS'] = home_goals_scored
        combined_df.at[idx, f'{home_team}_GC'] = home_goals_conceded
        combined_df.at[idx, f'{home_team}_Pts'] = home_points
        combined_df.at[idx, f'{home_team}_FormPtsStr'] = ''.join(home_form[-5:])
        combined_df.at[idx, f'{home_team}_FormPts'] = sum(3 if x == 'W' else 1 if x == 'D' else 0 for x in home_form[-5:])
        
        # Calculate win/loss streaks
        win_streak = 0
        loss_streak = 0
        for result in reversed(home_form):
            if result == 'W':
                win_streak += 1
                loss_streak = 0
            elif result == 'L':
                loss_streak += 1
                win_streak = 0
            else:
                win_streak = 0
                loss_streak = 0
            if win_streak >= 3:
                combined_df.at[idx, f'{home_team}_WinStreak3'] = 1
            if win_streak >= 5:
                combined_df.at[idx, f'{home_team}_WinStreak5'] = 1
            if loss_streak >= 3:
                combined_df.at[idx, f'{home_team}_LossStreak3'] = 1
            if loss_streak >= 5:
                combined_df.at[idx, f'{home_team}_LossStreak5'] = 1
    
    # Calculate away team features
    if not away_prev_matches.empty:
        away_goals_scored = 0
        away_goals_conceded = 0
        away_points = 0
        away_form = []
        
        for _, match in away_prev_matches.iterrows():
            if match['HomeTeam'] == away_team:
                away_goals_scored += match['FTHG']
                away_goals_conceded += match['FTAG']
                if match['FTR'] == 'H':
                    away_points += 3
                    away_form.append('W')
                elif match['FTR'] == 'D':
                    away_points += 1
                    away_form.append('D')
                else:
                    away_form.append('L')
            else:
                away_goals_scored += match['FTAG']
                away_goals_conceded += match['FTHG']
                if match['FTR'] == 'A':
                    away_points += 3
                    away_form.append('W')
                elif match['FTR'] == 'D':
                    away_points += 1
                    away_form.append('D')
                else:
                    away_form.append('L')
        
        combined_df.at[idx, f'{away_team}_GS'] = away_goals_scored
        combined_df.at[idx, f'{away_team}_GC'] = away_goals_conceded
        combined_df.at[idx, f'{away_team}_Pts'] = away_points
        combined_df.at[idx, f'{away_team}_FormPtsStr'] = ''.join(away_form[-5:])
        combined_df.at[idx, f'{away_team}_FormPts'] = sum(3 if x == 'W' else 1 if x == 'D' else 0 for x in away_form[-5:])
        
        # Calculate win/loss streaks
        win_streak = 0
        loss_streak = 0
        for result in reversed(away_form):
            if result == 'W':
                win_streak += 1
                loss_streak = 0
            elif result == 'L':
                loss_streak += 1
                win_streak = 0
            else:
                win_streak = 0
                loss_streak = 0
            if win_streak >= 3:
                combined_df.at[idx, f'{away_team}_WinStreak3'] = 1
            if win_streak >= 5:
                combined_df.at[idx, f'{away_team}_WinStreak5'] = 1
            if loss_streak >= 3:
                combined_df.at[idx, f'{away_team}_LossStreak3'] = 1
            if loss_streak >= 5:
                combined_df.at[idx, f'{away_team}_LossStreak5'] = 1

# Create final dataset with required columns
final_df = pd.DataFrame()

# Add basic match information
final_df['Date'] = combined_df['Date']
final_df['HomeTeam'] = combined_df['HomeTeam']
final_df['AwayTeam'] = combined_df['AwayTeam']
final_df['FTHG'] = combined_df['FTHG']
final_df['FTAG'] = combined_df['FTAG']
final_df['FTR'] = combined_df['FTR']

# Add historical features for home team
for team in teams:
    final_df[f'HTGS_{team}'] = combined_df[f'{team}_GS']
    final_df[f'HTGC_{team}'] = combined_df[f'{team}_GC']
    final_df[f'HTP_{team}'] = combined_df[f'{team}_Pts']
    final_df[f'HTFormPtsStr_{team}'] = combined_df[f'{team}_FormPtsStr']
    final_df[f'HTFormPts_{team}'] = combined_df[f'{team}_FormPts']
    final_df[f'HTWinStreak3_{team}'] = combined_df[f'{team}_WinStreak3']
    final_df[f'HTWinStreak5_{team}'] = combined_df[f'{team}_WinStreak5']
    final_df[f'HTLossStreak3_{team}'] = combined_df[f'{team}_LossStreak3']
    final_df[f'HTLossStreak5_{team}'] = combined_df[f'{team}_LossStreak5']

# Add historical features for away team
for team in teams:
    final_df[f'ATGS_{team}'] = combined_df[f'{team}_GS']
    final_df[f'ATGC_{team}'] = combined_df[f'{team}_GC']
    final_df[f'ATP_{team}'] = combined_df[f'{team}_Pts']
    final_df[f'ATFormPtsStr_{team}'] = combined_df[f'{team}_FormPtsStr']
    final_df[f'ATFormPts_{team}'] = combined_df[f'{team}_FormPts']
    final_df[f'ATWinStreak3_{team}'] = combined_df[f'{team}_WinStreak3']
    final_df[f'ATWinStreak5_{team}'] = combined_df[f'{team}_WinStreak5']
    final_df[f'ATLossStreak3_{team}'] = combined_df[f'{team}_LossStreak3']
    final_df[f'ATLossStreak5_{team}'] = combined_df[f'{team}_LossStreak5']

# Add goal difference and points difference
final_df['HTGD'] = final_df['FTHG'] - final_df['FTAG']
final_df['ATGD'] = final_df['FTAG'] - final_df['FTHG']

# Calculate points difference using team-specific columns
final_df['DiffPts'] = 0
final_df['DiffFormPts'] = 0

for idx, row in final_df.iterrows():
    home_team = row['HomeTeam']
    away_team = row['AwayTeam']
    
    # Get the points for the specific teams
    home_points = row[f'HTP_{home_team}']
    away_points = row[f'ATP_{away_team}']
    home_form_points = row[f'HTFormPts_{home_team}']
    away_form_points = row[f'ATFormPts_{away_team}']
    
    final_df.at[idx, 'DiffPts'] = home_points - away_points
    final_df.at[idx, 'DiffFormPts'] = home_form_points - away_form_points

# Add match statistics
final_df['HS'] = combined_df['HS']
final_df['AS'] = combined_df['AS']
final_df['HST'] = combined_df['HST']
final_df['AST'] = combined_df['AST']
final_df['HC'] = combined_df['HC']
final_df['AC'] = combined_df['AC']
final_df['HF'] = combined_df['HF']
final_df['AF'] = combined_df['AF']
final_df['HY'] = combined_df['HY']
final_df['AY'] = combined_df['AY']
final_df['HR'] = combined_df['HR']
final_df['AR'] = combined_df['AR']

# Add derived features
final_df['Goal_Difference'] = final_df['FTHG'] - final_df['FTAG']
final_df['Home_Shot_Accuracy'] = final_df['HST'] / final_df['HS'].replace(0, 1)
final_df['Away_Shot_Accuracy'] = final_df['AST'] / final_df['AS'].replace(0, 1)
final_df['Total_Cards'] = final_df['HY'] + final_df['AY'] + final_df['HR'] + final_df['AR']

# Save the final dataset
final_df.to_csv('preprocessed_combined_dataset_v2.csv', index=True) 