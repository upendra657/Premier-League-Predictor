import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Load data
df = pd.read_csv("final_dataset.csv")

# Date features
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df.drop(columns='Date', inplace=True)

# Remove leakage features
features_to_remove = ['FTHG', 'FTAG', 'HM1', 'HM2', 'HM3', 'HM4', 'HM5',
                      'AM1', 'AM2', 'AM3', 'AM4', 'AM5', 'HTFormPtsStr', 'ATFormPtsStr']
df.drop(columns=[col for col in features_to_remove if col in df.columns], inplace=True)

# Encode categorical features
categorical_cols = df.select_dtypes(include='object').columns.difference(['FTR'])
if not categorical_cols.empty:
    encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    encoded = pd.DataFrame(encoder.fit_transform(df[categorical_cols]),
                           columns=encoder.get_feature_names_out(categorical_cols),
                           index=df.index)
    df.drop(columns=categorical_cols, inplace=True)
    df = pd.concat([df, encoded], axis=1)

# Features and target
X = df.drop(columns='FTR')
y = df['FTR']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train model
model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

model.fit(X_train_scaled, y_train)

# Evaluate
train_acc = accuracy_score(y_train, model.predict(X_train_scaled))
test_acc = accuracy_score(y_test, model.predict(X_test_scaled))
print(f"\nTrain Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print("\nClassification Report:\n", classification_report(y_test, model.predict(X_test_scaled)))

# Confusion Matrix
cm = confusion_matrix(y_test, model.predict(X_test_scaled))
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=np.unique(y_test),
            yticklabels=np.unique(y_test))
plt.title(f"Random Forest Confusion Matrix\nAccuracy: {test_acc:.2%}")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("random_forest_confusion_matrix.png")
plt.show()

# Feature importance
feature_importance = pd.DataFrame({
    'Feature': X.columns,
    'Importance': model.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feature_importance.head(15))
plt.title("Top 15 Most Important Features")
plt.tight_layout()
plt.savefig("feature_importance.png")
plt.close()