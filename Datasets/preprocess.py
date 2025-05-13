import pandas as pd
from sklearn.preprocessing import OneHotEncoder

# Load dataset
df = pd.read_csv("final_testing_dataset.csv")

# One-hot encode HomeTeam and AwayTeam
encoder = OneHotEncoder(handle_unknown='ignore')
encoded = encoder.fit_transform(df[['HomeTeam', 'AwayTeam']]).toarray()
encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out(['HomeTeam', 'AwayTeam']))

# Drop original team columns and combine
df.drop(['HomeTeam', 'AwayTeam'], axis=1, inplace=True)
df = pd.concat([df.reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)

# Save to new file
df.to_csv("preprocessed_final_testing_dataset.csv", index=False)