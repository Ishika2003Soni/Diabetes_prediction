import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("diabetes.csv")
    return df

df = load_data()

# Split features and target
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Data scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Model training
model = LogisticRegression()
model.fit(X_train, y_train)

# Save model
joblib.dump((model, scaler), "diabetes_model.pkl")

# Accuracy
acc = accuracy_score(y_test, model.predict(X_test))

# Streamlit App
st.set_page_config(page_title="Diabetes Prediction App")

st.title("🩺 Diabetes Prediction using ML")
st.markdown("Enter your health details to check if you're at risk.")

# User input fields
pregnancies = st.number_input("Pregnancies", 0, 20, 1)
glucose = st.slider("Glucose", 0, 200, 120)
bp = st.slider("Blood Pressure", 0, 140, 70)
skin_thickness = st.slider("Skin Thickness", 0, 100, 20)
insulin = st.slider("Insulin", 0, 900, 79)
bmi = st.slider("BMI", 0.0, 70.0, 25.0)
dpf = st.slider("Diabetes Pedigree Function", 0.0, 2.5, 0.5)
age = st.slider("Age", 10, 100, 33)

# Predict button
if st.button("Predict"):
    input_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
    model, scaler = joblib.load("diabetes_model.pkl")
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)[0]
    
    if prediction == 1:
        st.error("⚠️ You are likely to have diabetes.")
    else:
        st.success("✅ You are not likely to have diabetes.")

st.caption(f"Model Accuracy: **{acc * 100:.2f}%**")
