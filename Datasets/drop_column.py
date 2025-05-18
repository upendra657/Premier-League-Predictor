import pandas as pd

# Load CSV (replace filename as needed)
df = pd.read_csv("final_dataset.csv")

# Drop the first column regardless of its name
df = df.iloc[:, 1:]

# Save back without index
df.to_csv("final_dataset.csv", index=False)

print("✅ Dropped first column and saved updated CSV.")