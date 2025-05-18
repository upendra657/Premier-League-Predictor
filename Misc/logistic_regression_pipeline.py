import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- Data Loading and Initial Processing ---
df = pd.read_csv("final_dataset.csv")
if 'Date' in df.columns:
    df['Date'] = pd.to_datetime(df['Date'])
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Day'] = df['Date'].dt.day
    df = df.drop('Date', axis=1)

# --- Remove Leaky Features ---
features_to_remove = [
    'FTHG', 'FTAG',  # Current match goals
    'HM1', 'HM2', 'HM3', 'HM4', 'HM5',  # Recent match results
    'AM1', 'AM2', 'AM3', 'AM4', 'AM5',
    'HTFormPtsStr', 'ATFormPtsStr',  # Form points strings
]
df = df.drop(columns=[col for col in features_to_remove if col in df.columns])

# --- Split Features and Target ---
X = df.drop(columns=["FTR"])
y = df["FTR"]

# --- Identify Column Types ---
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# --- Preprocessing Pipeline ---
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ]
)

# --- Full Pipeline ---
pipe = Pipeline([
    ('pre', preprocessor),
    ('clf', LogisticRegression(max_iter=2000, class_weight='balanced', solver='liblinear', random_state=42, C=0.01))
])

# --- Train/Test Split ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# --- Cross-validation ---
cv_scores = cross_val_score(pipe, X_train, y_train, cv=5)
print(f"Cross-validation scores: {cv_scores}")
print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

# --- Fit the Model ---
pipe.fit(X_train, y_train)

# --- Evaluation ---
y_pred = pipe.predict(X_test)
print("\nValidation Set Results:")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# --- Confusion Matrix ---
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["Home Win", "Away Win"],
            yticklabels=["Home Win", "Away Win"])
plt.title(f"Confusion Matrix - Validation Set\nOverall Accuracy: {accuracy_score(y_test, y_pred):.2%}", pad=20)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.show() 