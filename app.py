import streamlit as st
import numpy as np
import joblib

# ---------------------------
# PAGE CONFIG
# ---------------------------

st.set_page_config(
    page_title="Diabetes Prediction System",
    page_icon="🩺",
    layout="wide"
)

# ---------------------------
# LOAD MODEL
# ---------------------------

model = joblib.load("model/diabetes_model.pkl")
scaler = joblib.load("model/scaler.pkl")

# ---------------------------
# CUSTOM CSS
# ---------------------------

st.markdown("""
<style>

.main {
    background-color: #0E1117;
}

h1, h2, h3 {
    color: #00E5FF;
}

.stButton > button {
    background-color: #00E5FF;
    color: black;
    font-weight: bold;
    border-radius: 10px;
    height: 50px;
    width: 100%;
}

.footer {
    text-align: center;
    padding: 20px;
    font-size: 16px;
}

</style>
""", unsafe_allow_html=True)

# ---------------------------
# SIDEBAR
# ---------------------------

st.sidebar.title("👨‍💻 Developer")

st.sidebar.write("""
**Prajyot Yesankar**

Aspiring Data Analyst & Data Science Enthusiast

### Skills
- Python
- SQL
- Power BI
- Machine Learning
- Data Analytics
""")

# ---------------------------
# HEADER
# ---------------------------

st.title("🩺 Diabetes Prediction System")

st.markdown("""
Predict the likelihood of diabetes using Machine Learning and Logistic Regression.
""")

st.markdown("---")

# ---------------------------
# ABOUT PROJECT
# ---------------------------

st.header("📌 About Project")

st.write("""
The Diabetes Prediction System predicts whether a patient is likely to have diabetes
based on medical indicators such as glucose level, BMI, insulin level,
blood pressure, age, and family history.

### Machine Learning Algorithm
- Logistic Regression

### Dataset
- Pima Indians Diabetes Dataset

### Project Features
- Data Cleaning
- Exploratory Data Analysis (EDA)
- Feature Engineering
- Logistic Regression
- Model Evaluation
- Streamlit Web Application
""")

st.markdown("---")

# ---------------------------
# DATASET INSIGHTS
# ---------------------------

st.header("📊 Dataset Insights")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Patients", "768")

with col2:
    st.metric("Diabetic", "268")

with col3:
    st.metric("Non-Diabetic", "500")

with col4:
    st.metric("Model Accuracy", "78%")

st.markdown("---")

# ---------------------------
# PATIENT FORM
# ---------------------------

st.header("📝 Enter Patient Information")

col1, col2 = st.columns(2)

with col1:
    pregnancies = st.number_input("Pregnancies", min_value=0)
    glucose = st.number_input("Glucose", min_value=0)
    blood_pressure = st.number_input("Blood Pressure", min_value=0)
    skin_thickness = st.number_input("Skin Thickness", min_value=0)

with col2:
    insulin = st.number_input("Insulin", min_value=0)
    bmi = st.number_input("BMI", min_value=0.0)
    dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0)
    age = st.number_input("Age", min_value=1)

# ---------------------------
# PREDICTION
# ---------------------------

if st.button("🔮 Predict Diabetes Risk"):

    data = np.array([[
        pregnancies,
        glucose,
        blood_pressure,
        skin_thickness,
        insulin,
        bmi,
        dpf,
        age
    ]])

    data_scaled = scaler.transform(data)

    prediction = model.predict(data_scaled)

    probability = model.predict_proba(data_scaled)

    st.markdown("---")

    st.header("📈 Prediction Result")

    if prediction[0] == 1:

        st.error("⚠️ High Risk of Diabetes")

        st.write(
            f"### Probability of Diabetes: {round(probability[0][1] * 100, 2)}%"
        )

        st.warning("""
        Please consult a healthcare professional for further evaluation.
        """)

    else:

        st.success("✅ Low Risk of Diabetes")

        st.write(
            f"### Probability of Diabetes: {round(probability[0][1] * 100, 2)}%"
        )

        st.info("""
        Continue maintaining a healthy lifestyle and regular health checkups.
        """)

st.markdown("---")

# ---------------------------
# WORKFLOW
# ---------------------------

st.header("⚙️ Project Workflow")

st.write("""
1. Data Collection
2. Data Cleaning
3. Exploratory Data Analysis
4. Feature Scaling
5. Logistic Regression
6. Model Evaluation
7. Streamlit Deployment
""")

st.markdown("---")

# ---------------------------
# LOGISTIC REGRESSION
# ---------------------------

st.header("📚 About Logistic Regression")

st.write("""
Logistic Regression is a Machine Learning Classification Algorithm
used to predict binary outcomes.

In this project:

- 0 = No Diabetes
- 1 = Diabetes

The model calculates the probability of diabetes and then classifies
the patient based on the prediction threshold.
""")

st.markdown("---")

# ---------------------------
# ABOUT DEVELOPER
# ---------------------------

st.header("🙋 About Developer")

st.write("""
Hi, I'm **Prajyot Yesankar**.

B.Tech Computer Science & Design Graduate.

Interested in:

- Data Analytics
- Data Science
- SQL
- Power BI
- Python
- Machine Learning

I enjoy building real-world projects that solve practical business and healthcare problems.
""")

st.markdown("---")

# ---------------------------
# CONTACT
# ---------------------------
st.markdown("""
<div style="text-align:center; font-size:20px;">
<a href="https://www.linkedin.com/in/prajyot-yesankar-79215b258/" target="_blank">💼 LinkedIn</a>
&nbsp;&nbsp;|&nbsp;&nbsp;
<a href="https://github.com/yesankarprajyot123" target="_blank">💻 GitHub</a>
&nbsp;&nbsp;|&nbsp;&nbsp;
<a href="mailto:yesankarprajyot@gmail.com">📧 Email</a>
</div>
""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center; font-size:18px;'>
❤️ Developed by <b>Prajyot Yesankar</b>
<br><br>
Python | SQL | Power BI | Machine Learning | Data Analytics
</div>
""", unsafe_allow_html=True)