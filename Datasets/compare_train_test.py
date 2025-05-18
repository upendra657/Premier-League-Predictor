import pandas as pd

# Load datasets
train = pd.read_csv("final_dataset.csv")
test = pd.read_csv("final_testing_ddataset.csv")

# Remove target and date columns if present
train_cols = set(train.columns) - {'FTR', 'Date'}
test_cols = set(test.columns) - {'FTR', 'Date'}

# Compare
print("Missing in test:", train_cols - test_cols)
print("Extra in test:", test_cols - train_cols)
print("Common features:", len(train_cols & test_cols))