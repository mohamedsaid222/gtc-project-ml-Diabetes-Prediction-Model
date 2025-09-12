import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# ======================
# 1. Load trained models (using joblib)
# ======================
logreg_model = joblib.load("final_logreg_model.pkl")
rf_model = joblib.load("final_rf_model.pkl")
xgb_model = joblib.load("final_xgb_model.pkl")

models = {
    "Logistic Regression": logreg_model,
    "Random Forest": rf_model,
    "XGBoost": xgb_model
}

# Feature names (from dataset)
features = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"
]

# ======================
# 2. App title
# ======================
st.title("🩺 Diabetes Prediction App")
st.write("Enter patient details to predict the likelihood of diabetes and explore feature importance.")

# ======================
# 3. User inputs (Sidebar)
# ======================
with st.sidebar:
    st.header("Patient Data Input")
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
    age = st.number_input("Age", min_value=1, max_value=120, value=33)

# DataFrame for input
input_data = pd.DataFrame([[
    pregnancies, glucose, blood_pressure, skin_thickness,
    insulin, bmi, dpf, age
]], columns=features)

# ======================
# 4. Model selection
# ======================
model_choice = st.selectbox("Choose Model", list(models.keys()))
model = models[model_choice]

# ======================
# 5. Prediction
# ======================
if st.button("Predict"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ The model predicts this patient is likely to have Diabetes.")
    else:
        st.success("✅ The model predicts this patient is NOT likely to have Diabetes.")

# ======================
# 6. Feature Importance
# ======================
st.subheader("🔎 Feature Importance Analysis")

if model_choice == "Random Forest":
    rf_importance = model.named_steps["clf"].feature_importances_
    df_rf = pd.DataFrame({
        "Feature": features,
        "Importance": rf_importance
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=df_rf, ax=ax)
    ax.set_title("Random Forest Feature Importance")
    st.pyplot(fig)

elif model_choice == "XGBoost":
    xgb_importance = model.named_steps["clf"].feature_importances_
    df_xgb = pd.DataFrame({
        "Feature": features,
        "Importance": xgb_importance
    }).sort_values("Importance", ascending=False)

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.barplot(x="Importance", y="Feature", data=df_xgb, ax=ax)
    ax.set_title("XGBoost Feature Importance")
    st.pyplot(fig)

else:
    st.info("Feature importance is not available for Logistic Regression.")
