import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load dataset
df = pd.read_csv("cleaned_2021_2022.csv")
df = pd.read_csv("cleaned_2020_2021.csv")

# Convert date column
df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

# Encode categorical columns
le = LabelEncoder()
df["HomeTeam"] = le.fit_transform(df["HomeTeam"])
df["AwayTeam"] = le.fit_transform(df["AwayTeam"])
df["FTR"] = le.fit_transform(df["FTR"])

# Feature Engineering
df["Goal_Difference"] = df["FTHG"] - df["FTAG"]
df["Home_Shot_Accuracy"] = df["HST"] / df["HS"].replace(0, 1)
df["Away_Shot_Accuracy"] = df["AST"] / df["AS"].replace(0, 1)
df["Total_Cards"] = df["HY"] + df["AY"] + df["HR"] + df["AR"]

# Fill missing values
df.fillna(0, inplace=True)

# Save
df.to_csv("preprocessed_dataset.csv", index=False)
