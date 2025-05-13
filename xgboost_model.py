import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set matplotlib to show plots in a window
plt.ion()  # Turn on interactive mode

# Get the current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the datasets with correct paths
print("Loading datasets...")
train_data = pd.read_csv(os.path.join(current_dir, 'final_dataset.csv'))
test_data = pd.read_csv(os.path.join(current_dir, 'final_testing_dataset.csv'))

# Basic data cleaning and feature extraction
print("\nProcessing dates and extracting features...")
if 'Date' in train_data.columns:
    train_data['Date'] = pd.to_datetime(train_data['Date'])
    train_data['Year'] = train_data['Date'].dt.year
    train_data['Month'] = train_data['Date'].dt.month
    train_data['Day'] = train_data['Date'].dt.day
    train_data = train_data.drop('Date', axis=1)

if 'Date' in test_data.columns:
    test_data['Date'] = pd.to_datetime(test_data['Date'])
    test_data['Year'] = test_data['Date'].dt.year
    test_data['Month'] = test_data['Date'].dt.month
    test_data['Day'] = test_data['Date'].dt.day
    test_data = test_data.drop('Date', axis=1)

# Remove features that would cause data leakage
print("\nRemoving features that could cause data leakage...")
features_to_remove = [
    'FTHG', 'FTAG',  # Current match goals
    'HM1', 'HM2', 'HM3', 'HM4', 'HM5',  # Recent match results
    'AM1', 'AM2', 'AM3', 'AM4', 'AM5',
    'HTFormPtsStr', 'ATFormPtsStr',  # Form points strings
]

train_data = train_data.drop(columns=[col for col in features_to_remove if col in train_data.columns])
test_data = test_data.drop(columns=[col for col in features_to_remove if col in test_data.columns])

# One-hot encode HomeTeam and AwayTeam
print("\nEncoding team names (HomeTeam, AwayTeam)...")
categorical_cols = []
if 'HomeTeam' in train_data.columns:
    categorical_cols.append('HomeTeam')
if 'AwayTeam' in train_data.columns:
    categorical_cols.append('AwayTeam')

if categorical_cols:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    # Fit on train, transform both train and test
    encoded_train = pd.DataFrame(
        encoder.fit_transform(train_data[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=train_data.index
    )
    encoded_test = pd.DataFrame(
        encoder.transform(test_data[categorical_cols]),
        columns=encoder.get_feature_names_out(categorical_cols),
        index=test_data.index
    )
    # Drop original columns and concat encoded
    train_data = train_data.drop(columns=categorical_cols)
    test_data = test_data.drop(columns=categorical_cols)
    train_data = pd.concat([train_data, encoded_train], axis=1)
    test_data = pd.concat([test_data, encoded_test], axis=1)

# Prepare features and target
X_train = train_data.drop(columns=['FTR'])
y_train = train_data['FTR']
X_test = test_data.drop(columns=['FTR'])
y_test = test_data['FTR']

# Find common features between train and test sets
common_features = list(set(X_train.columns) & set(X_test.columns))
print(f"\nNumber of common features: {len(common_features)}")

# Use only common features
X_train = X_train[common_features]
X_test = X_test[common_features]

# Print dataset information
print("\nDataset Information:")
print(f"Number of features: {X_train.shape[1]}")
print(f"Number of training samples: {X_train.shape[0]}")
print(f"Number of test samples: {X_test.shape[0]}")
print("\nClass Distribution in Training Set:")
print(y_train.value_counts())
print("\nClass Distribution in Test Set:")
print(y_test.value_counts())

# Scale features
print("\nScaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert back to DataFrame to maintain column names
X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)

# Print unique values in both datasets
print("\nUnique values in training set:", y_train.unique())
print("Unique values in test set:", y_test.unique())

# Instead of filtering out unseen labels, we'll map them to the closest seen label
print("\nMapping unseen labels to closest seen labels...")
train_labels = set(y_train.unique())
test_labels = set(y_test.unique())
unseen_labels = test_labels - train_labels

# Create a mapping for unseen labels
label_mapping = {
    'A': 'NH',  # Away win maps to Non-home
    'D': 'NH',  # Draw maps to Non-home
    'H': 'H',   # Home win stays as Home win
    'NH': 'NH', # Non-home stays as Non-home
    'NA': 'NH'  # Non-away maps to Non-home
}

# Apply the mapping to test labels
y_test_mapped = y_test.map(label_mapping)

# Report the mapping results
print("\nLabel mapping results:")
for old_label, new_label in label_mapping.items():
    count = (y_test == old_label).sum()
    if count > 0:
        print(f"{old_label} -> {new_label}: {count} matches")

# Encode the target variable
le = LabelEncoder()
le.fit(pd.concat([pd.Series(y_train), pd.Series(y_test_mapped)]).unique())

# Transform both datasets
y_train_enc = le.transform(y_train)
y_test_enc = le.transform(y_test_mapped)

# Check number of classes in training set
n_classes = len(np.unique(y_train_enc))

# Always use binary classification
xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    min_child_weight=1,
    gamma=0,
    subsample=0.8,
    colsample_bytree=0.8,
    objective='binary:logistic',
    random_state=42
)

# Train the model
print("\nTraining XGBoost model...")
xgb_model.fit(X_train_scaled, y_train_enc)

# Make predictions
y_pred_prob = xgb_model.predict_proba(X_test_scaled)
y_pred = (y_pred_prob[:, 1] > 0.5).astype(int)

# Calculate accuracy
accuracy = accuracy_score(y_test_enc, y_pred)
print(f"\nModel Accuracy: {accuracy:.4f}")

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

# Create confusion matrix
plt.figure(figsize=(10, 8))
cm = confusion_matrix(y_test_enc, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title(f'Confusion Matrix\nAccuracy: {accuracy:.2%}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(current_dir, 'xgboost_confusion_matrix.png'))
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': xgb_model.feature_importances_
})
feature_importance = feature_importance.sort_values('importance', ascending=False)

# Plot feature importance
plt.figure(figsize=(12, 8))
sns.barplot(x='importance', y='feature', data=feature_importance.head(20))
plt.title('Top 20 Feature Importance')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'xgboost_feature_importance.png'))

# Save the model
xgb_model.save_model(os.path.join(current_dir, 'xgboost_model.json'))

# Print top 10 most important features
print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Keep the plots open until user closes them
plt.show(block=True)

# Create a new version of the test set with relabeled targets for full evaluation
full_test_path = os.path.join(current_dir, 'final_testing_dataset_xgb_full.csv')
test_data_full = test_data.copy()
test_data_full['FTR'] = test_data_full['FTR'].replace({'A': 'NH', 'D': 'NH'})
test_data_full.to_csv(full_test_path, index=False)
print(f"\nCreated new test set for XGBoost: {full_test_path}")

# Load the new test set
X_test_full = test_data_full.drop(columns=['FTR'])
y_test_full = test_data_full['FTR']

# Use only common features
X_test_full = X_test_full[[col for col in X_test_full.columns if col in X_train.columns]]
X_test_full = X_test_full[X_train.columns]  # Ensure order

# Scale features
X_test_full_scaled = scaler.transform(X_test_full)
X_test_full_scaled = pd.DataFrame(X_test_full_scaled, columns=X_test_full.columns)

# Encode the target variable for the full test set
y_test_full_enc = le.transform(y_test_full)

# Predict on the full test set
y_pred_full = xgb_model.predict(X_test_full_scaled)

# Confusion matrix for the full test set
cm_full = confusion_matrix(y_test_full_enc, y_pred_full)
print(f"\nConfusion matrix for the full test set (shape: {cm_full.shape}):\n{cm_full}")
print(f"Expected confusion matrix size: {n_classes}x{n_classes} = {n_classes**2} values.")

# Plot and save the confusion matrix for the full test set
plt.figure(figsize=(10, 8))
sns.heatmap(cm_full, annot=True, fmt='d', cmap='Blues',
            xticklabels=le.classes_,
            yticklabels=le.classes_)
plt.title(f'Confusion Matrix (All Test Data)\nAccuracy: {accuracy_score(y_test_full_enc, y_pred_full):.2%}')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig(os.path.join(current_dir, 'xgboost_confusion_matrix_full.png'))
plt.close()
