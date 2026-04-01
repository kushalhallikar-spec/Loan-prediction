import streamlit as st
import numpy as np
import pickle

# Load model & columns
model = pickle.load(open('loan_model.pkl', 'rb'))
columns = pickle.load(open('columns.pkl', 'rb'))

st.title("💳 Loan Approval Prediction")

# Inputs
gender = st.selectbox("Gender (0=Female, 1=Male)", [0,1])
married = st.selectbox("Married (0=No, 1=Yes)", [0,1])
dependents = st.selectbox("Dependents", [0,1,2,3])
education = st.selectbox("Education (0=Graduate, 1=Not Graduate)", [0,1])
self_employed = st.selectbox("Self Employed (0=No, 1=Yes)", [0,1])

app_income = st.number_input("Applicant Income")
co_income = st.number_input("Coapplicant Income")

loan_amount = st.number_input("Loan Amount")
loan_term = st.number_input("Loan Term")

credit_history = st.selectbox("Credit History (0/1)", [0,1])
property_area = st.selectbox("Property Area (0/1/2)", [0,1,2])

# Prediction
if st.button("Predict"):
    
    input_dict = {
        'Gender': gender,
        'Married': married,
        'Dependents': dependents,
        'Education': education,
        'Self_Employed': self_employed,
        'ApplicantIncome': app_income,
        'CoapplicantIncome': co_income,
        'LoanAmount': loan_amount,
        'Loan_Amount_Term': loan_term,
        'Credit_History': credit_history,
        'Property_Area': property_area
    }

    # Match correct column order
    data = [input_dict[col] for col in columns]
    data = np.array([data])

    prediction = model.predict(data)

    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Rejected ❌")