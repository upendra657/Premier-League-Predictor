import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load and prepare the dataset
print("Loading dataset...")
df = pd.read_csv("final_dataset.csv")

# Basic data cleaning and feature extraction
print("\nProcessing dates and extracting features...")
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df = df.drop('Date', axis=1)

# Remove features that would cause data leakage
# We keep historical features but remove current match information
features_to_remove = [
    'FTHG', 'FTAG',  # Current match goals
    'HM1', 'HM2', 'HM3', 'HM4', 'HM5',  # Recent match results
    'AM1', 'AM2', 'AM3', 'AM4', 'AM5',
    'HTFormPtsStr', 'ATFormPtsStr',  # Form points strings
]

print("\nRemoving features that could cause data leakage...")
df = df.drop(columns=[col for col in features_to_remove if col in df.columns])

# Process categorical variables
print("\nProcessing categorical variables...")
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
if 'FTR' in categorical_cols:
    categorical_cols.remove('FTR')

# One-hot encode categorical variables
if categorical_cols:
    print("Encoding team names...")
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded_cols = pd.DataFrame(encoder.fit_transform(df[categorical_cols]))
    encoded_cols.columns = encoder.get_feature_names_out(categorical_cols)
    
    df = df.drop(categorical_cols, axis=1)
    df = pd.concat([df, encoded_cols], axis=1)

# Prepare features and target
X = df.drop(columns=["FTR"])
y = df["FTR"]

# Print dataset information
print("\nDataset Information:")
print(f"Number of features: {X.shape[1]}")
print(f"Number of samples: {X.shape[0]}")
print("\nClass Distribution:")
print(y.value_counts())

# Split the data
print("\nSplitting data into train and test sets...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
print("Scaling features...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
print("\nTraining random forest model...")
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=None,
    class_weight='balanced',
    random_state=42
)

# Cross-validation
print("Performing cross-validation...")
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Fit the model
model.fit(X_train_scaled, y_train)

# Feature importance
print("\nAnalyzing feature importance...")
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by="Importance", ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10))

# Model evaluation
print("\nEvaluating model performance...")
train_pred = model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train, train_pred)
print(f"\nTraining Accuracy: {train_accuracy:.4f}")

print("\nTraining Classification Report:")
print(classification_report(y_train, train_pred))

# Validation set predictions
y_pred = model.predict(X_test_scaled)

print("\nValidation Set Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix for Validation Set
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_pred, labels=np.unique(y_test))
labels_sorted = np.unique(y_test)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=labels_sorted,
            yticklabels=labels_sorted)
val_accuracy = accuracy_score(y_test, y_pred)
plt.title(f"Confusion Matrix - Validation Set\nAccuracy: {val_accuracy:.2%}", pad=20)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig('validation_confusion_matrix.png')
plt.show()
plt.close()

# Print detailed metrics
print("\nDetailed Performance Analysis:")
print(f"Total predictions: {cm.sum()}")
print(f"Correct predictions: {cm.diagonal().sum()}")
print(f"Overall accuracy: {val_accuracy:.2%}")

# Plot feature importance
print("\nPlotting feature importance...")
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Top 15 Most Important Features')
plt.tight_layout()
plt.savefig('feature_importance.png')
plt.close()

# Test on final dataset
print("\nTesting on final_testing_dataset.csv...")
test_df = pd.read_csv("final_testing_dataset.csv")

# Process test data
if 'Date' in test_df.columns:
    test_df['Date'] = pd.to_datetime(test_df['Date'])
    test_df['Year'] = test_df['Date'].dt.year
    test_df['Month'] = test_df['Date'].dt.month
    test_df['Day'] = test_df['Date'].dt.day
    test_df = test_df.drop('Date', axis=1)

# Remove same features as training data
test_df = test_df.drop(columns=[col for col in features_to_remove if col in test_df.columns])

# Get the target variable
y_test_final = test_df["FTR"]

# Prepare test data - skip categorical encoding since it's already done
X_test_final = test_df.drop(columns=["FTR", "HomeTeam", "AwayTeam"])

# Align final test set with training features
missing_cols = set(X_train.columns) - set(X_test_final.columns)
for col in missing_cols:
    X_test_final[col] = 0  # Add missing columns with default value

# Drop extra columns not seen during training
extra_cols = set(X_test_final.columns) - set(X_train.columns)
X_test_final.drop(columns=extra_cols, inplace=True)

# Ensure columns are in the same order
X_test_final = X_test_final[X_train.columns]

# Scale and predict
X_test_final_scaled = scaler.transform(X_test_final)
y_pred_final = model.predict(X_test_final_scaled)

# Evaluate final results
print("\nFinal Test Results:")
print(f"Accuracy: {accuracy_score(y_test_final, y_pred_final):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_final, y_pred_final))