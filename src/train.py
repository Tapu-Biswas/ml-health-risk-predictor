import pandas as pd
import joblib
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

df = pd.read_csv('data/diabetes_clean.csv')

# Save the medians used for imputation so the app can apply the same logic to live input
raw_df = pd.read_csv('data/diabetes.csv')
cols_with_missing = ['glucose', 'blood_pressure', 'skin_thickness', 'insulin', 'bmi']
medians = {}
for col in cols_with_missing:
    valid_values = raw_df[raw_df[col] != 0][col]
    medians[col] = valid_values.median()

os.makedirs('models', exist_ok=True)
with open('models/impute_medians.json', 'w') as f:
    json.dump(medians, f, indent=2)
print("Saved imputation medians:", medians)

X = df.drop('outcome', axis=1)
y = df['outcome']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

model = RandomForestClassifier(n_estimators=200, random_state=42)
model.fit(X_train_scaled, y_train)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
}
print(classification_report(y_test, y_pred))
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print(metrics)

joblib.dump(model, 'models/rf_model.pkl')
joblib.dump(scaler, 'models/scaler.pkl')
with open('models/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)