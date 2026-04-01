import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Load dataset
url = "https://raw.githubusercontent.com/dsrscientist/DSData/master/loan_prediction.csv"
df = pd.read_csv(url)
df.drop('Loan_ID', axis=1, inplace=True)

# Handle missing values
for col in df.select_dtypes(include='number'):
    df[col].fillna(df[col].median(), inplace=True)

for col in df.select_dtypes(include='object'):
    df[col].fillna(df[col].mode()[0], inplace=True)

# Encode categorical data
le = LabelEncoder()
for col in df.select_dtypes(include='object').columns:
    df[col] = le.fit_transform(df[col])

# Features & Target
X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

# Save column names (IMPORTANT)
pickle.dump(X.columns, open('columns.pkl', 'wb'))

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier(max_depth=5)
model.fit(X_train, y_train)

# Save model
pickle.dump(model, open('loan_model.pkl', 'wb'))

print("Train:", model.score(X_train, y_train))
print("Test:", model.score(X_test, y_test))