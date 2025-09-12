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
st.title("🧑‍⚕️ Diabetes Prediction App")
st.write("Enter patient details to predict the likelihood of diabetes and explore feature importance and EDA visualizations.")

# ======================
# 3. User inputs (Center, not Sidebar)
# ======================
st.subheader("🩸 Patient Data Entry")

col1, col2, col3 = st.columns(3)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
    glucose = st.number_input("Glucose", min_value=0, max_value=300, value=120)
    blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=200, value=70)

with col2:
    skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
    insulin = st.number_input("Insulin", min_value=0, max_value=900, value=80)
    bmi = st.number_input("BMI", min_value=0.0, max_value=70.0, value=25.0)

with col3:
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
st.subheader("🧬 Choose Prediction Model")
model_choice = st.selectbox(
    "Model",
    list(models.keys()),
    index=list(models.keys()).index("Random Forest")  # Default = Random Forest
)
model = models[model_choice]

# ======================
# 5. Prediction + Feature Importance
# ======================
if st.button("🧾 Predict"):
    prediction = model.predict(input_data)[0]
    proba = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(input_data)[0][1]

    if prediction == 1:
        if proba is not None:
            st.error(f"⚠️ Patient is likely to have Diabetes (Probability: {proba:.2%})")
        else:
            st.error("⚠️ Patient is likely to have Diabetes.")
    else:
        if proba is not None:
            st.success(f"✅ Patient is NOT likely to have Diabetes (Probability: {proba:.2%})")
        else:
            st.success("✅ Patient is NOT likely to have Diabetes.")

    # Show Feature Importance only after prediction
    st.subheader("📊 Feature Importance Analysis")

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
        st.info("ℹ️ Feature importance is not available for Logistic Regression.")

    # ======================
    # 6. Extra EDA Visualizations
    # ======================
    st.subheader("📉 Exploratory Data Analysis (EDA)")

    # Load dataset (for visualization only)
    df = pd.read_csv("diabetes.csv")

    # Correlation Heatmap
    st.write("### 🔗 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

    # Distribution of Features
    st.write("### 📈 Feature Distributions")
    selected_feature = st.selectbox("Choose a feature to visualize:", features)
    fig, ax = plt.subplots(figsize=(6, 4))
    sns.histplot(df[selected_feature], kde=True, bins=30, ax=ax)
    ax.set_title(f"Distribution of {selected_feature}")
    st.pyplot(fig)
