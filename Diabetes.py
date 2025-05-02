import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, mean_squared_error, recall_score, precision_score, confusion_matrix, f1_score

import xgboost as xgb
from xgboost import XGBClassifier

data = pd.read_csv(r"diabetes.csv")
data.isnull().sum()

data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']] = data[['Glucose','BloodPressure','SkinThickness','Insulin','BMI']].replace(0 , np.nan) 
data.isnull().sum()

columns = ['Glucose','BloodPressure','SkinThickness','Insulin','BMI']
for column in columns:
    temp = data.groupby('Outcome')[column].median()
    data.loc[(data['Outcome'] == 0) & (data[column].isna()) , column ] = temp[0]
    data.loc[(data['Outcome'] == 1) & (data[column].isna()) , column ] = temp[1]

numerical_columns = data.columns.to_list()[:-1]

for col in numerical_columns:
    Q1 = data[col].quantile(0.25)  # First quartile (25%)
    Q3 = data[col].quantile(0.75)  # Third quartile (75%)
    IQR = Q3 - Q1  # Interquartile range
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    data = data[(data[col] > lower_bound) & (data[col] < upper_bound)]

X = data.drop('Outcome', axis=1)
y = data['Outcome']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 44)

xgb_train = xgb.DMatrix(X_train, label = y_train)
xgb_test = xgb.DMatrix(X_test, label = y_test)

model_xgb = XGBClassifier(learning_rate=0.6, random_state=42)
model_xgb.fit(X_train, y_train)

y_predict_xgb = model_xgb.predict(X_test)

accuracy=accuracy_score( y_test, y_predict_xgb)
recall = recall_score(y_test, y_predict_xgb)
precision = precision_score(y_test, y_predict_xgb)
mse= mean_squared_error( y_test, y_predict_xgb)

print(f"Accuracy: {accuracy}")
print(f"Recall: {recall}")
print(f"Precision: {precision}")
print(f"F1: {f1_score(y_test, y_predict_xgb)}")
print(f"Mean squared error: {mse}")
print(f"Confusion_Matrix: \n {confusion_matrix(y_test, y_predict_xgb)}")

import streamlit as st
import joblib
import numpy as np

# Load the trained model
model = joblib.load("diabetes_model.pkl")

# Streamlit UI
st.title("Diabetes Prediction App")
st.write("Enter patient details below to predict diabetes.")

# Input fields
pregnancies = st.number_input("Pregnancies", min_value=0, max_value=20, value=1)
glucose = st.number_input("Glucose Level", min_value=0, max_value=200, value=100)
blood_pressure = st.number_input("Blood Pressure", min_value=0, max_value=150, value=70)
skin_thickness = st.number_input("Skin Thickness", min_value=0, max_value=100, value=20)
insulin = st.number_input("Insulin Level", min_value=0, max_value=500, value=80)
bmi = st.number_input("BMI", min_value=0.0, max_value=50.0, value=25.0)
dpf = st.number_input("Diabetes Pedigree Function", min_value=0.0, max_value=3.0, value=0.5)
age = st.number_input("Age", min_value=0, max_value=120, value=30)

# Predict button
if st.button("Predict"):
    input_data = np.array([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, dpf, age]).reshape(1, -1)
    prediction = model.predict(input_data)[0]
    
    if prediction == 1:
        st.error("The model predicts **Diabetes** (Positive Case).")
    else:
        st.success("The model predicts **No Diabetes** (Negative Case).")
