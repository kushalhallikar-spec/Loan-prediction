```python
import streamlit as st
import pickle
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Loan Predictor", page_icon="🏦", layout="centered")

# -------------------- LOAD MODEL --------------------
model = pickle.load(open("loan_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# -------------------- TITLE --------------------
st.markdown("<h1 style='text-align: center;'>🏦 Loan Approval Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Check your loan eligibility instantly</p>", unsafe_allow_html=True)

st.markdown("---")

# -------------------- INPUT SECTION --------------------
st.subheader("📋 Enter Applicant Details")

col1, col2 = st.columns(2)

with col1:
    gender = st.selectbox("Gender", ["Male", "Female"])
    married = st.selectbox("Married", ["Yes", "No"])
    education = st.selectbox("Education", ["Graduate", "Not Graduate"])
    applicant_income = st.number_input("Applicant Income", min_value=0)

with col2:
    dependents = st.selectbox("Dependents", ["0", "1", "2", "3+"])
    self_employed = st.selectbox("Self Employed", ["Yes", "No"])
    loan_amount = st.number_input("Loan Amount", min_value=0)
    credit_history = st.selectbox("Credit History (Good=1)", [1, 0])

property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

st.markdown("---")

# -------------------- PREDICTION --------------------
if st.button("🔍 Predict Loan Status"):
    
    # Example input (modify based on your encoding)
    input_dict = {
        "Gender": 1 if gender == "Male" else 0,
        "Married": 1 if married == "Yes" else 0,
        "Education": 1 if education == "Graduate" else 0,
        "Self_Employed": 1 if self_employed == "Yes" else 0,
        "ApplicantIncome": applicant_income,
        "LoanAmount": loan_amount,
        "Credit_History": credit_history,
        "Dependents": 0 if dependents == "0" else 1,
    }

    # Convert to model input
    data = [input_dict.get(col, 0) for col in columns]
    data = np.array(data).reshape(1, -1)

    # Loader
    with st.spinner("Predicting..."):
        prediction = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

    st.markdown("---")

    # -------------------- OUTPUT --------------------
    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.info(f"Approval Probability: {prob*100:.2f}%")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made by Kushal 🚀</p>", unsafe_allow_html=True)

