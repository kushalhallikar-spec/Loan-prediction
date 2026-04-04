# 🏦 Loan Approval Prediction

> A Machine Learning project that predicts whether a loan application will be approved based on applicant demographics and financial history — built as an end-to-end ML pipeline with Streamlit deployment.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Deployed-Streamlit-red?style=flat-square&logo=streamlit)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-Classification-orange?style=flat-square)
![Notebook](https://img.shields.io/badge/Jupyter-Notebook-yellow?style=flat-square&logo=jupyter)

---

## 🧠 Overview

Loan approval decisions involve multiple financial and demographic factors. This project builds a classification model to predict approval outcomes, helping financial institutions automate and standardise the screening process.

---

## 📊 Dataset

- **Features:** Gender, Marital Status, Education, Income, Loan Amount, Credit History, Property Area
- **Target variable:** `Loan_Status` — Y (Approved) / N (Rejected)
- **Preprocessing:** Missing value imputation, label encoding, feature scaling

---

## ⚙️ ML Pipeline

```
Raw Data
   │
   ▼
Exploratory Data Analysis (EDA)
   │
   ▼
Missing Value Handling & Encoding
   │
   ▼
Feature Scaling
   │
   ▼
Model Training & Evaluation
   │
   ▼
Streamlit Deployment
```

---

## 📈 Results

| Model | Accuracy |
|---|---|
| Logistic Regression | ~80% |
| Random Forest | ~82% |
| Best Model (deployed) | ~82% |

---

## 🔍 Key Insights

- **Credit history** is the single most important feature for loan approval
- Applicants with higher combined income have significantly better approval rates
- Graduate applicants are approved more often than non-graduates
- Property area (urban vs rural) has a noticeable impact on approval

---

## 🛠️ Tech Stack

- Python, Pandas, NumPy
- Scikit-learn (Logistic Regression, Random Forest)
- Matplotlib, Seaborn
- Streamlit (Deployment)

---

## 🚀 Run Locally

```bash
git clone https://github.com/kushalhallikar-spec/Loan-prediction.git
cd Loan-prediction
pip install -r requirements.txt
streamlit run app.py
```

---

## 🔮 Future Improvements

- [ ] Add XGBoost and compare all models with cross-validation
- [ ] Add SHAP explainability — show why a loan was rejected
- [ ] Build a full form-based UI for realistic applicant input
- [ ] Add confidence score display alongside prediction

---

## 👨‍💻 Author

**Kushal Hallikar**
Aspiring Machine Learning Engineer

[![GitHub](https://img.shields.io/badge/GitHub-kushalhallikar--spec-181717?style=flat-square&logo=github)](https://github.com/kushalhallikar-spec)

---

⭐ If you found this useful, consider giving it a star!
