import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
import os

# === CONFIGURATION ===
label_map = {'H': 0, 'D': 1, 'A': 2}
label_order = ['H', 'D', 'A']

# === Load Data ===
print("Loading datasets...")
current_dir = os.path.dirname(os.path.abspath(__file__))
train_data = pd.read_csv(os.path.join(current_dir, 'final_dataset.csv'))
test_data = pd.read_csv(os.path.join(current_dir, 'final_dataset.csv'))

# === Process Dates ===
for df in [train_data, test_data]:
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
        df['Year'] = df['Date'].dt.year
        df['Month'] = df['Date'].dt.month
        df['Day'] = df['Date'].dt.day
        df.drop(columns='Date', inplace=True)

# === Drop Leakage Columns ===
leak_cols = ['FTHG', 'FTAG', 'HM1','HM2','HM3','HM4','HM5',
             'AM1','AM2','AM3','AM4','AM5', 'HTFormPtsStr','ATFormPtsStr']
train_data.drop(columns=[c for c in leak_cols if c in train_data.columns], inplace=True)
test_data.drop(columns=[c for c in leak_cols if c in test_data.columns], inplace=True)

# === One-hot Encode Teams ===
team_cols = ['HomeTeam', 'AwayTeam']
encoder = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

train_encoded = pd.DataFrame(
    encoder.fit_transform(train_data[team_cols]),
    columns=encoder.get_feature_names_out(team_cols),
    index=train_data.index
)
test_encoded = pd.DataFrame(
    encoder.transform(test_data[team_cols]),
    columns=encoder.get_feature_names_out(team_cols),
    index=test_data.index
)

train_data.drop(columns=team_cols, inplace=True)
test_data.drop(columns=team_cols, inplace=True)
train_data = pd.concat([train_data, train_encoded], axis=1)
test_data = pd.concat([test_data, test_encoded], axis=1)

# === Features and Labels ===
X_train = train_data.drop(columns=['FTR'])
y_train = train_data['FTR'].astype(str)
X_test = test_data.drop(columns=['FTR'])
y_test = test_data['FTR'].astype(str)

# === Label Mapping ===
y_train_enc = y_train.map(label_map)
y_test_enc = y_test.map(label_map)

# === Drop Invalid Labels ===
train_mask = y_train_enc.notna()
test_mask = y_test_enc.notna()

X_train = X_train[train_mask]
y_train_enc = y_train_enc[train_mask]
X_test = X_test[test_mask]
y_test_enc = y_test_enc[test_mask]

# === Sanity Check ===
if X_test.shape[0] == 0:
    raise ValueError("❌ No valid samples in test set after filtering. Check the 'FTR' column in the test dataset.")

# === Align Columns ===
common_cols = list(set(X_train.columns) & set(X_test.columns))
X_train = X_train[common_cols]
X_test = X_test[common_cols]

# === Scaling ===
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# === Train XGBoost ===
model = xgb.XGBClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=5,
    objective='multi:softprob',
    num_class=3,
    use_label_encoder=False,
    random_state=42
)

print("Training model...")
model.fit(X_train_scaled, y_train_enc)

# === Evaluation ===
train_pred = model.predict(X_train_scaled)
test_pred = model.predict(X_test_scaled)
train_acc = accuracy_score(y_train_enc, train_pred)
test_acc = accuracy_score(y_test_enc, test_pred)

print(f"\n✅ Training Accuracy: {train_acc:.4f}")
print(f"✅ Testing Accuracy: {test_acc:.4f}")

# === Reports ===
print("\nClassification Report (Test Set):")
print(classification_report(y_test_enc, test_pred, target_names=label_order))

# === Confusion Matrix ===
cm = confusion_matrix(y_test_enc, test_pred, labels=[0, 1, 2])
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_order, yticklabels=label_order)
plt.title(f'Confusion Matrix\nTrain Acc: {train_acc:.2%} | Test Acc: {test_acc:.2%}')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.tight_layout()
plt.savefig(os.path.join(current_dir, 'xgboost_confusion_matrix.png'))
plt.show()