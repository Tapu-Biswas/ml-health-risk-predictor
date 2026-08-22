import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

# Load model, scaler, and data
model = joblib.load('models/rf_model.pkl')
scaler = joblib.load('models/scaler.pkl')
df = pd.read_csv('data/diabetes_clean.csv')
X = df.drop('outcome', axis=1)

# Scale features the same way training did
X_scaled = scaler.transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)

# Build SHAP explainer for tree-based models
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_scaled_df)

os.makedirs('models/plots', exist_ok=True)

# --- Global explanation: which features matter most overall ---
plt.figure()
shap.summary_plot(shap_values[:, :, 1], X_scaled_df, show=False)
plt.tight_layout()
plt.savefig('models/plots/shap_summary.png', dpi=150)
plt.close()
print("Saved global SHAP summary plot.")

# --- Local explanation: why ONE specific patient was flagged ---
# Pick the first patient predicted as high-risk, as an example
sample_idx = 0
plt.figure()
shap.plots.waterfall(
    shap.Explanation(
        values=shap_values[sample_idx, :, 1],
        base_values=explainer.expected_value[1],
        data=X_scaled_df.iloc[sample_idx],
        feature_names=X.columns.tolist()
    ),
    show=False
)
plt.tight_layout()
plt.savefig('models/plots/shap_waterfall_sample.png', dpi=150)
plt.close()
print("Saved local SHAP waterfall plot for sample patient.")

# Save the explainer itself for reuse in the Streamlit app
joblib.dump(explainer, 'models/shap_explainer.pkl')
print("Saved SHAP explainer.")