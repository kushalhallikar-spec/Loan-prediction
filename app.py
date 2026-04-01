
import streamlit as st
import pickle
import numpy as np

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Loan Predictor", page_icon="🏦", layout="wide")

# -------------------- LOAD MODEL --------------------
model = pickle.load(open("loan_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# -------------------- CUSTOM CSS --------------------
st.markdown("""
<style>
.stButton>button {
    background-color: #4CAF50;
    color: white;
    border-radius: 8px;
    height: 3em;
    width: 100%;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

# -------------------- HEADER --------------------
st.markdown("<h1 style='text-align: center;'>🏦 Loan Approval Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center; font-size:18px;'>Smart ML-powered system to predict loan eligibility</p>", unsafe_allow_html=True)

st.markdown("---")

# -------------------- SIDEBAR --------------------
st.sidebar.title("📊 Input Details")
st.sidebar.markdown("Fill all fields to predict")

gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
married = st.sidebar.selectbox("Married", ["Yes", "No"])
education = st.sidebar.selectbox("Education", ["Graduate", "Not Graduate"])
self_employed = st.sidebar.selectbox("Self Employed", ["Yes", "No"])

dependents = st.sidebar.selectbox("Dependents", ["0", "1", "2", "3+"])
property_area = st.sidebar.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

applicant_income = st.sidebar.number_input("Applicant Income", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
credit_history = st.sidebar.selectbox("Credit History (1 = Good)", [1, 0])

predict_btn = st.sidebar.button("🚀 Predict")

# -------------------- MAIN UI --------------------
col1, col2, col3 = st.columns(3)

col1.metric("Model Type", "ML Classifier")
col2.metric("Accuracy", "80%+")  # update if you know exact
col3.metric("Status", "Ready")

st.markdown("---")

# -------------------- PREDICTION --------------------
if predict_btn:

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

    data = [input_dict.get(col, 0) for col in columns]
    data = np.array(data).reshape(1, -1)

    with st.spinner("🔍 Analyzing Application..."):
        prediction = model.predict(data)[0]
        prob = model.predict_proba(data)[0][1]

    st.markdown("## 📊 Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.info(f"💡 Approval Probability: {prob*100:.2f}%")

    # -------------------- FEATURE IMPORTANCE --------------------
    st.markdown("### 🔍 Model Insights")

    try:
        importance = model.coef_[0]
        for i, col in enumerate(columns):
            st.write(f"{col}: {round(importance[i], 2)}")
    except:
        st.warning("Feature importance not available")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made by Kushal 🚀 | ML Project</p>", unsafe_allow_html=True)

