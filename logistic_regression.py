import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Load dataset
df = pd.read_csv("final_dataset.csv")

# Extract date features
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop(columns='Date', inplace=True)

# Remove leakage-prone features
features_to_remove = ['FTHG', 'FTAG', 'HM1', 'HM2', 'HM3', 'HM4', 'HM5',
                      'AM1', 'AM2', 'AM3', 'AM4', 'AM5', 'HTFormPtsStr', 'ATFormPtsStr']
df.drop(columns=[col for col in features_to_remove if col in df.columns], inplace=True)

# One-hot encode categorical features
categorical_cols = df.select_dtypes(include='object').columns.difference(['FTR'])
if not categorical_cols.empty:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded = pd.DataFrame(encoder.fit_transform(df[categorical_cols]),
                           columns=encoder.get_feature_names_out(categorical_cols),
                           index=df.index)
    df.drop(columns=categorical_cols, inplace=True)
    df = pd.concat([df, encoded], axis=1)

# Split features and target
X = df.drop(columns='FTR')
y = df['FTR']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression Model (with regularization to prevent overfitting)
model = LogisticRegression(
    max_iter=2000, class_weight='balanced',
    solver='liblinear', C=0.01, random_state=42
)

# Cross-validation
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# Train model
model.fit(X_train_scaled, y_train)

# Feature importance
importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': np.abs(model.coef_[0])
}).sort_values('Importance', ascending=False)

# Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
test_pred = model.predict(X_test_scaled)
test_acc = accuracy_score(y_test, test_pred)

print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, test_pred))

# Save model
joblib.dump(model, "logistic_regression_model.joblib")
print("✅ Model saved as 'logistic_regression_model.joblib'")

# Save feature schema for Streamlit use
# After the model is trained and just before saving:
if not categorical_cols.empty:
    feature_names = list(X.columns.difference(categorical_cols)) + list(encoder.get_feature_names_out(categorical_cols))
else:
    feature_names = list(X.columns)

with open("logistic_regression_features.txt", "w") as f:
    for col in feature_names:
        f.write(f"{col}\n")

print("✅ Model and feature schema saved.")

# Confusion Matrix
cm = confusion_matrix(y_test, test_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.title(f"Confusion Matrix\nTrain Acc: {train_acc:.2%} | Test Acc: {test_acc:.2%}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("logistic_regression_confusion_matrix.png")
plt.show()

# Feature importance plot
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=importance.head(15))
plt.title("Top 15 Most Important Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()

