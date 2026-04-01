```python
import streamlit as st
import pickle
import numpy as np
import pandas as pd

# -------------------- PAGE CONFIG --------------------
st.set_page_config(page_title="Loan Predictor", page_icon="🏦", layout="wide")

# -------------------- LOAD MODEL --------------------
model = pickle.load(open("loan_model.pkl", "rb"))
columns = pickle.load(open("columns.pkl", "rb"))

# -------------------- LIGHT UI BUTTON STYLE --------------------
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
st.markdown("<p style='text-align: center;'>Predict loan eligibility using Machine Learning</p>", unsafe_allow_html=True)

st.markdown("---")

# -------------------- SIDEBAR INPUT --------------------
st.sidebar.title("📊 Applicant Details")

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

# -------------------- METRICS --------------------
col1, col2, col3 = st.columns(3)
col1.metric("Model Type", "ML Classifier")
col2.metric("Accuracy", "80%+")
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

    # ---------------- RESULT ----------------
    st.markdown("## 📊 Prediction Result")

    if prediction == 1:
        st.success("✅ Loan Approved")
    else:
        st.error("❌ Loan Rejected")

    st.info(f"💡 Approval Probability: {prob*100:.2f}%")

    # ---------------- FEATURE IMPORTANCE ----------------
    st.markdown("### 🔍 Model Insights")

    try:
        if hasattr(model, "coef_"):
            importance = model.coef_[0]
        elif hasattr(model, "feature_importances_"):
            importance = model.feature_importances_
        else:
            raise Exception()

        importance_df = pd.DataFrame({
            "Feature": columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(importance_df.set_index("Feature"))

    except:
        st.warning("Feature importance not available for this model")

    # ---------------- EXPLANATION ----------------
    st.markdown("### 🧠 Why this prediction?")

    reasons = []

    if credit_history == 0:
        reasons.append("❌ Poor credit history")

    if applicant_income < 2500:
        reasons.append("💰 Low applicant income")

    if loan_amount > 300:
        reasons.append("📉 High loan amount")

    if self_employed == "No":
        reasons.append("👤 Not self employed")

    if dependents != "0":
        reasons.append("👨‍👩‍👧 Has dependents")

    if prediction == 0:
        if reasons:
            st.error("Loan rejected due to:")
            for r in reasons:
                st.write(r)
        else:
            st.warning("Loan rejected due to multiple factors")
    else:
        st.success("Loan approved due to strong financial profile ✅")

# -------------------- FOOTER --------------------
st.markdown("---")
st.markdown("<p style='text-align: center;'>Made by Kushal 🚀</p>", unsafe_allow_html=True)
```
