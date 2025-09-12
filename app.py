# ====================================
# Diabetes Prediction App - Streamlit
# ====================================

import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------
# Load saved models
# -------------------------------
models = {
    "Logistic Regression": joblib.load("final_logreg_model.pkl"),
    "Random Forest": joblib.load("final_rf_model.pkl"),
    "XGBoost": joblib.load("final_xgb_model.pkl")
}

# -------------------------------
# Gauge function
# -------------------------------
def plot_gauge(probability, prediction):
    fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw={'projection': 'polar'})
    angle = probability * np.pi
    theta = np.linspace(0, np.pi, 200)
    r = np.ones_like(theta)
    ax.plot(theta, r, color="black", linewidth=3)
    ax.fill_between(theta, 0, 1, color="lightgray", alpha=0.3)
    ax.plot([angle, angle], [0, 1],
            color="red" if prediction == 1 else "green", linewidth=4)
    ax.set_ylim(0, 1)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_axis_off()
    label = "Diabetic" if prediction == 1 else "Not Diabetic"
    ax.set_title(f"Prob = {probability:.2f}\n{label}", fontsize=11, weight="bold")
    return fig

# -------------------------------
# Streamlit Layout
# -------------------------------
st.set_page_config(page_title="Diabetes Prediction", page_icon="🩺", layout="wide")
st.title("🩺 Diabetes Prediction App")

tab1, tab2 = st.tabs(["🔍 Prediction", "📊 Model Insights"])

# -------------------------------
# Tab 1: Prediction
# -------------------------------
with tab1:
    st.markdown("Predict the likelihood of diabetes using **Machine Learning models**. "
                "Select a model, enter patient data, and get instant predictions.")

    st.sidebar.header("⚙️ Settings")
    model_choice = st.sidebar.selectbox("Select Model", list(models.keys()))

    if "prediction_log" not in st.session_state:
        st.session_state.prediction_log = pd.DataFrame(columns=[
            "Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DPF","Age","Prediction","Probability","Model"
        ])

    st.header("👤 Patient Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        pregnancies = st.number_input("Pregnancies", 0, 20, 2)
        bp = st.number_input("Blood Pressure", 0, 150, 72)
        bmi = st.number_input("BMI", 0.0, 70.0, 33.6)
    with col2:
        glucose = st.number_input("Glucose", 0, 200, 120)
        skin = st.number_input("Skin Thickness", 0, 100, 20)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.627)
    with col3:
        insulin = st.number_input("Insulin", 0, 900, 80)
        age = st.number_input("Age", 1, 120, 30)

    if st.button("🔍 Predict Diabetes"):
        input_data = pd.DataFrame([{
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DPF": dpf,
            "Age": age
        }])

        model = models[model_choice]
        prediction = model.predict(input_data)[0]
        proba = model.predict_proba(input_data)[0][1]

        st.subheader("📊 Prediction Result")
        if prediction == 1:
            st.error(f"⚠️ Patient is **Diabetic** (Probability = {proba:.2f})")
        else:
            st.success(f"✅ Patient is **Not Diabetic** (Probability = {proba:.2f})")

        fig = plot_gauge(proba, prediction)
        st.pyplot(fig)

        new_entry = {
            "Pregnancies": pregnancies,
            "Glucose": glucose,
            "BloodPressure": bp,
            "SkinThickness": skin,
            "Insulin": insulin,
            "BMI": bmi,
            "DPF": dpf,
            "Age": age,
            "Prediction": "Diabetic" if prediction==1 else "Not Diabetic",
            "Probability": round(proba, 2),
            "Model": model_choice
        }
        st.session_state.prediction_log = pd.concat(
            [st.session_state.prediction_log, pd.DataFrame([new_entry])],
            ignore_index=True
        )

    if not st.session_state.prediction_log.empty:
        st.subheader("📑 Prediction Log")
        st.dataframe(st.session_state.prediction_log, use_container_width=True)

        csv = st.session_state.prediction_log.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="⬇️ Download Log as CSV",
            data=csv,
            file_name="prediction_log.csv",
            mime="text/csv"
        )

# -------------------------------
# Tab 2: Model Insights
# -------------------------------
with tab2:
    st.markdown("Explore model insights such as **Feature Importance**.")

    features = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
                "Insulin","BMI","DPF","Age"]

    rf_model = models["Random Forest"]
    xgb_model = models["XGBoost"]

    # Feature Importance plots (fixed with zip)
    st.subheader("🔎 Feature Importance")
    fig, axes = plt.subplots(1,2, figsize=(14,6))

    rf_importance = rf_model.feature_importances_
    df_rf = pd.DataFrame(list(zip(features, rf_importance)),
                         columns=["Feature", "Importance"]).sort_values("Importance", ascending=False)
    sns.barplot(x="Importance", y="Feature", data=df_rf, ax=axes[0], palette="viridis")
    axes[0].set_title("Random Forest Feature Importance")

    xgb_importance = xgb_model.feature_importances_
    df_xgb = pd.DataFrame(list(zip(features, xgb_importance)),
                          columns=["Feature", "Importance"]).sort_values("Importance", ascending=False)
    sns.barplot(x="Importance", y="Feature", data=df_xgb, ax=axes[1], palette="plasma")
    axes[1].set_title("XGBoost Feature Importance")

    st.pyplot(fig)


