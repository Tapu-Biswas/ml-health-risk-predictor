import streamlit as st
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="Diabetes Risk Predictor", layout="centered")

@st.cache_resource
def load_artifacts():
    model = joblib.load('models/rf_model.pkl')
    scaler = joblib.load('models/scaler.pkl')
    explainer = joblib.load('models/shap_explainer.pkl')
    return model, scaler, explainer

model, scaler, explainer = load_artifacts()

st.title("🩺 Diabetes Risk Predictor")
st.write("Enter patient health metrics below to estimate diabetes risk, with an explanation of why.")

col1, col2 = st.columns(2)
with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
with col2:
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    diabetes_pedigree = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=30)

if st.button("Predict Risk"):
    input_df = pd.DataFrame([{
        'pregnancies': pregnancies,
        'glucose': glucose,
        'blood_pressure': blood_pressure,
        'skin_thickness': skin_thickness,
        'insulin': insulin,
        'bmi': bmi,
        'diabetes_pedigree': diabetes_pedigree,
        'age': age
    }])

    input_scaled = scaler.transform(input_df)
    proba = model.predict_proba(input_scaled)[0][1]
    prediction = model.predict(input_scaled)[0]

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High risk — predicted probability: {proba:.1%}")
    else:
        st.success(f"✅ Low risk — predicted probability: {proba:.1%}")

    st.subheader("Why this prediction?")
    input_scaled_df = pd.DataFrame(input_scaled, columns=input_df.columns)
    shap_values = explainer.shap_values(input_scaled_df)

    fig, ax = plt.subplots()
    shap.plots.waterfall(
        shap.Explanation(
            values=shap_values[0, :, 1],
            base_values=explainer.expected_value[1],
            data=input_scaled_df.iloc[0],
            feature_names=input_df.columns.tolist()
        ),
        show=False
    )
    st.pyplot(fig)

st.divider()
st.caption("Model: RandomForestClassifier trained on the Pima Indians Diabetes Dataset. Explanations via SHAP. Not a substitute for medical diagnosis.")