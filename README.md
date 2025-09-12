🧑‍⚕️ Diabetes Prediction Model

-This project is a Machine Learning application built with Streamlit that predicts the likelihood of diabetes based on patient health data.
It includes:

-Multiple ML models (Logistic Regression, Random Forest, XGBoost).

-Interactive web interface with Streamlit.

-Feature importance analysis & Exploratory Data Analysis (EDA).

-Probability-based predictions (not just binary).

----------
📂 Project Structure
gtc-project-ml-Diabetes-Prediction-Model/
│
├── app.py                  # Main Streamlit app
├── train_models.py         # Script to train and save ML models
├── diabetes.csv            # Dataset (Pima Indians Diabetes Database)
├── final_logreg_model.pkl  # Saved Logistic Regression model
├── final_rf_model.pkl      # Saved Random Forest model
├── final_xgb_model.pkl     # Saved XGBoost model
├── requirements.txt        # Dependencies for Streamlit Cloud
└── README.md               # Project documentation

--------
📊 Dataset

We used the Pima Indians Diabetes Database, which contains health data for female patients of Pima Indian heritage.

Features included:

Pregnancies

Glucose

Blood Pressure

Skin Thickness

Insulin

BMI

Diabetes Pedigree Function

Age

Target:

0 → No Diabetes

1 → Diabetes

🤖 Machine Learning Models

We trained and evaluated three models:

Logistic Regression → baseline model.

Random Forest → ensemble model with feature importance.

XGBoost → gradient boosting model with high accuracy.

Training (train_models.py)

The dataset was split into train/test sets.

Models were trained and saved with joblib (.pkl).

Best parameters were tuned for Random Forest & XGBoost.

🌐 Streamlit App Features
1️⃣ Input Patient Data

User enters medical details (Pregnancies, Glucose, etc.) in a centered form (not sidebar).

2️⃣ Model Selection

Dropdown to choose the ML model.

Default = Random Forest.

3️⃣ Prediction

Displays if the patient is likely to have diabetes.

Shows probability score (e.g., "21% likely").

4️⃣ Feature Importance

For Random Forest & XGBoost → bar chart of most important features.

Logistic Regression → Info message (not supported).

5️⃣ Exploratory Data Analysis (EDA)

Correlation Heatmap (feature correlations).

Feature Distribution plots (histograms with KDE).

📈 Example Screenshots

(You can add screenshots here once the app is running on Streamlit Cloud)

Home Page

Prediction Result

Feature Importance

EDA Charts

☁️ Deployment

The app is deployed on Streamlit Cloud:
👉 https://diabetes-prediction-model-vdevbrvrjt78e42iddpum.streamlit.app
------------


Link

🛠 Requirements

requirements.txt includes:

streamlit==1.36.0
scikit-learn==1.6.1
xgboost==2.0.3
joblib
pandas
numpy
matplotlib
seaborn

🚀 Future Improvements

Add model comparison charts (Accuracy, Recall, F1-score).

Integrate SHAP values for explainability.

Deploy with Docker for portability.

Add user authentication for private use in clinics.

👨‍💻 Author

Mohamed Said

💼 Machine Learning Engineer

🌐 https://www.linkedin.com/in/mohamed-said-b3bb52222
