import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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
print("\nTraining logistic regression model...")
model = LogisticRegression(
    max_iter=2000,
    class_weight='balanced',
    solver='liblinear',
    C=0.01,  # Strong regularization to prevent overfitting
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
    'Importance': np.abs(model.coef_[0])
})
feature_importance = feature_importance.sort_values('Importance', ascending=False)

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

# Confusion Matrix
print("\nGenerating confusion matrix...")
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))

# Calculate metrics
total = cm.sum()
correct_predictions = cm.diagonal().sum()
accuracy = correct_predictions / total

# Calculate class metrics
class_metrics = []
for i, label in enumerate(["Home Win", "Not Home Win"]):
    true_positives = cm[i, i]
    false_positives = cm[:, i].sum() - true_positives
    false_negatives = cm[i, :].sum() - true_positives
    
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    class_metrics.append({
        'label': label,
        'precision': precision,
        'recall': recall,
        'f1': f1
    })

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Home Win", "Not Home Win"],
            yticklabels=["Home Win", "Not Home Win"])

plt.title(f"Confusion Matrix - Validation Set\nOverall Accuracy: {accuracy:.2%}", pad=20)
plt.xlabel("Predicted")
plt.ylabel("Actual")

# Add metrics text
metrics_text = "\n".join(
    f"{m['label']}:\nPrecision: {m['precision']:.2%}\nRecall: {m['recall']:.2%}\nF1-score: {m['f1']:.2%}"
    for m in class_metrics
)

plt.figtext(1.02, 0.5, metrics_text, fontsize=10, va='center')
plt.tight_layout()
plt.show()

# Print detailed metrics
print("\nDetailed Performance Analysis:")
print(f"Total predictions: {total}")
print(f"Correct predictions: {correct_predictions}")
print(f"Overall accuracy: {accuracy:.2%}")

for metrics in class_metrics:
    print(f"\n{metrics['label']}:")
    print(f"Precision: {metrics['precision']:.2%}")
    print(f"Recall: {metrics['recall']:.2%}")
    print(f"F1-score: {metrics['f1']:.2%}")

# Plot feature importance
print("\nPlotting feature importance...")
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title('Top 15 Most Important Features')
plt.tight_layout()
plt.show()

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

# Process categorical variables
test_categorical_cols = test_df.select_dtypes(include=['object']).columns.tolist()
if 'FTR' in test_categorical_cols:
    test_categorical_cols.remove('FTR')

# Ensure test data has same columns as training data
if test_categorical_cols:
    for col in categorical_cols:
        if col not in test_categorical_cols:
            test_df[col] = 0
    
    test_df = test_df.reindex(columns=df.columns)
    
    test_encoded_cols = pd.DataFrame(encoder.transform(test_df[categorical_cols]))
    test_encoded_cols.columns = encoder.get_feature_names_out(categorical_cols)
    
    test_df = test_df.drop(categorical_cols, axis=1)
    test_df = pd.concat([test_df, test_encoded_cols], axis=1)

# Prepare test data
X_test_final = test_df.drop(columns=["FTR"])
y_test_final = test_df["FTR"]

# Scale and predict
X_test_final_scaled = scaler.transform(X_test_final)
y_pred_final = model.predict(X_test_final_scaled)

# Evaluate final results
print("\nFinal Test Results:")
print(f"Accuracy: {accuracy_score(y_test_final, y_pred_final):.4f}")
print("\nClassification Report:")
print(classification_report(y_test_final, y_pred_final))

# Final confusion matrix
cm_final = confusion_matrix(y_test_final, y_pred_final)
plt.figure(figsize=(6, 4))
sns.heatmap(cm_final, annot=True, fmt="d", cmap="Blues", 
            xticklabels=["Home Win", "Not Home Win"],
            yticklabels=["Home Win", "Not Home Win"])
plt.title("Confusion Matrix - Test Dataset")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()