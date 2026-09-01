# 🩺 Diabetes Risk Predictor

A machine learning web app that estimates diabetes risk from patient health metrics, with transparent, per-prediction explanations powered by SHAP.

**🔗 Live demo:** https://ml-health-risk-predictor.streamlit.app

## What it does

Enter basic health metrics (glucose, BMI, age, etc.) and get:
- A risk prediction (high/low) with probability
- A SHAP waterfall chart showing exactly which factors drove that specific prediction
- Automatic handling of missing/unknown fields
- A confidence flag when the input falls outside what the model was trained on

## Why this project

Built to explore two things that matter in applied ML: **model interpretability** (a prediction without an explanation isn't very useful in a healthcare context) and **honest limitations** (a model that's silently wrong is worse than one that flags its own uncertainty).

## Dataset

[Pima Indians Diabetes Dataset](https://archive.ics.uci.edu/ml/index.php) — 768 patient records, 8 features. Known limitation: this dataset only contains female patients from one population, so the model's outputs shouldn't be generalized beyond that context.

**Note on data quality**: this dataset encodes missing values as `0` for glucose, blood pressure, skin thickness, insulin, and BMI (biologically impossible values). These were identified during EDA and handled via median imputation during both training and inference.

## Model & Performance

- **Algorithm**: RandomForestClassifier (scikit-learn)
- **Accuracy**: 74%
- **ROC-AUC**: 0.82

These numbers are consistent with published benchmarks on this dataset (typically 73-82% accuracy across studies) — this is a genuinely hard, small, real-world dataset, and reported numbers well above this range on tabular baselines are usually a sign of data leakage rather than a better model.

**Known limitation**: recall on the positive (diabetic) class is 56% — the model misses a meaningful share of true positive cases. In a real screening context, this would need to be weighed carefully against false-positive costs.

## Explainability

SHAP (SHapley Additive exPlanations) is used for both global and per-prediction interpretability:
- Global: glucose, BMI, and age are the model's strongest predictors — consistent with established clinical risk factors
- Local: every prediction includes a waterfall chart breaking down exactly which inputs pushed the risk score up or down, and by how much

## Known limitations

- Trained on a single-population, historical dataset — not validated for broader demographic use
- Glucose has an outsized influence on predictions relative to other factors (disclosed in-app)
- Small dataset size (768 records) limits how much complexity the model can reliably learn

## Tech stack

Python · pandas · scikit-learn · SHAP · Streamlit · matplotlib

## Running locally

```bash
git clone https://github.com/Tapu-Biswas/ml-health-risk-predictor.git
cd ml-health-risk-predictor
python -m venv venv
venv\Scripts\activate  # Windows
pip install -r requirements.txt
python src/train.py
python src/explain.py
streamlit run app.py
```