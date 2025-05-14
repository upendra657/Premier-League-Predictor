import pandas as pd
import numpy as np
import os

def transform_test_dataset():
    # Load datasets
    current_dir = os.path.dirname(os.path.abspath(__file__))
    train_data = pd.read_csv(os.path.join(current_dir, 'final_dataset.csv'))
    test_data = pd.read_csv(os.path.join(current_dir, 'final_testing_dataset_xgb_full.csv'))
    
    # Create a dictionary to store all columns
    transformed_dict = {}
    
    # Extract basic match information
    transformed_dict['Date'] = pd.to_datetime(test_data['Year'].astype(str) + '-' + 
                                            test_data['Month'].astype(str) + '-' + 
                                            test_data['Day'].astype(str))
    
    # Extract team names from one-hot encoded columns
    home_team_cols = [col for col in test_data.columns if col.startswith('HomeTeam_')]
    away_team_cols = [col for col in test_data.columns if col.startswith('AwayTeam_')]
    
    # Get team names from the first occurrence of 1 in each row
    transformed_dict['HomeTeam'] = test_data[home_team_cols].idxmax(axis=1).str.replace('HomeTeam_', '')
    transformed_dict['AwayTeam'] = test_data[away_team_cols].idxmax(axis=1).str.replace('AwayTeam_', '')
    
    # Map FTR values
    transformed_dict['FTR'] = test_data['FTR'].map({'0': 'H', '1': 'D', '2': 'A'})
    
    # Initialize team-specific features with default values
    teams = train_data['HomeTeam'].unique()
    team_features = {}
    
    for team in teams:
        # Home team features
        team_features[f'HTGS_{team}'] = 0
        team_features[f'HTGC_{team}'] = 0
        team_features[f'HTP_{team}'] = 0
        team_features[f'HTFormPtsStr_{team}'] = 'MMMMM'
        team_features[f'HTFormPts_{team}'] = 0
        team_features[f'HTWinStreak3_{team}'] = 0
        team_features[f'HTWinStreak5_{team}'] = 0
        team_features[f'HTLossStreak3_{team}'] = 0
        team_features[f'HTLossStreak5_{team}'] = 0
        
        # Away team features
        team_features[f'ATGS_{team}'] = 0
        team_features[f'ATGC_{team}'] = 0
        team_features[f'ATP_{team}'] = 0
        team_features[f'ATFormPtsStr_{team}'] = 'MMMMM'
        team_features[f'ATFormPts_{team}'] = 0
        team_features[f'ATWinStreak3_{team}'] = 0
        team_features[f'ATWinStreak5_{team}'] = 0
        team_features[f'ATLossStreak3_{team}'] = 0
        team_features[f'ATLossStreak5_{team}'] = 0
    
    # Add team features to the dictionary
    transformed_dict.update(team_features)
    
    # Add match statistics
    transformed_dict.update({
        'HS': test_data['HS'].fillna(0),
        'AS': test_data['AS'].fillna(0),
        'HST': test_data['HST'].fillna(0),
        'AST': test_data['AST'].fillna(0),
        'HC': test_data['HC'].fillna(0),
        'AC': test_data['AC'].fillna(0),
        'HF': test_data['HF'].fillna(0),
        'AF': test_data['AF'].fillna(0),
        'HY': test_data['HY'].fillna(0),
        'AY': test_data['AY'].fillna(0),
        'HR': test_data['HR'].fillna(0),
        'AR': test_data['AR'].fillna(0)
    })
    
    # Add derived features
    transformed_dict.update({
        'Goal_Difference': test_data['Goal_Difference'].fillna(0),
        'Home_Shot_Accuracy': test_data['Home_Shot_Accuracy'].fillna(0),
        'Away_Shot_Accuracy': test_data['Away_Shot_Accuracy'].fillna(0),
        'Total_Cards': test_data['Total_Cards'].fillna(0)
    })
    
    # Add date features
    transformed_dict.update({
        'Year': test_data['Year'],
        'Month': test_data['Month'],
        'Day': test_data['Day']
    })
    
    # Create DataFrame from dictionary
    transformed_test = pd.DataFrame(transformed_dict)
    
    # Calculate goal differences
    transformed_test['HTGD'] = transformed_test['Goal_Difference']
    transformed_test['ATGD'] = -transformed_test['Goal_Difference']
    
    # Calculate points difference
    for idx, row in transformed_test.iterrows():
        home_team = row['HomeTeam']
        away_team = row['AwayTeam']
        
        if home_team and away_team:
            home_points = row[f'HTP_{home_team}']
            away_points = row[f'ATP_{away_team}']
            home_form_points = row[f'HTFormPts_{home_team}']
            away_form_points = row[f'ATFormPts_{away_team}']
            
            transformed_test.at[idx, 'DiffPts'] = home_points - away_points
            transformed_test.at[idx, 'DiffFormPts'] = home_form_points - away_form_points
    
    # Save the transformed dataset
    output_path = os.path.join(current_dir, 'final_testing_dataset_transformed.csv')
    transformed_test.to_csv(output_path, index=False)
    print(f"Transformed test dataset saved to: {output_path}")
    
    # Verify the transformation
    print("\nVerifying transformation:")
    print(f"Number of columns in training data: {len(train_data.columns)}")
    print(f"Number of columns in transformed test data: {len(transformed_test.columns)}")
    print("\nColumns in training data but not in transformed test data:")
    missing_cols = set(train_data.columns) - set(transformed_test.columns)
    print(missing_cols if missing_cols else "None")
    print("\nColumns in transformed test data but not in training data:")
    extra_cols = set(transformed_test.columns) - set(train_data.columns)
    print(extra_cols if extra_cols else "None")

if __name__ == "__main__":
    transform_test_dataset() 