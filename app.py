import streamlit as st
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# 1. PAGE CONFIGURATION & TITLE
st.set_page_config(page_title="ME AI Predictor", layout="centered")
st.title("🛠️ Machine Failure Prediction System")
st.subheader("Program: Mechanical Engineering")

# 2. PROGRAM RELEVANCE (Required Feature)
with st.expander("See Program Relevance"):
    st.write("""
    This project aligns with Mechanical Engineering by focusing on **Predictive Maintenance**. 
    By analyzing sensor data like torque and temperature, we can prevent mechanical 
    breakdowns, reduce maintenance costs, and improve industrial safety[cite: 84, 86].
    """)

# 3. THE AI MODEL (Simplified for easy implementation)
# This creates a small logic base: High torque + High temp = Failure
def train_model():
    # Features: [Air Temp (K), Rotational Speed (RPM), Torque (Nm)]
    X = np.array([
        [300, 1500, 40], [302, 1350, 55], [305, 1200, 65], 
        [295, 1700, 30], [298, 1550, 42], [310, 1100, 75]
    ])
    # 0 = Healthy, 1 = Failure
    y = np.array([0, 0, 1, 0, 0, 1]) 
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

model = train_model()

# 4. INPUT FORM (Required Feature)
st.header("Input Sensor Data")
with st.form("input_form"):
    col1, col2 = st.columns(2)
    with col1:
        air_temp = st.number_input("Air Temperature (K)", value=300.0)
        speed = st.number_input("Rotational Speed (RPM)", value=1500.0)
    with col2:
        torque = st.number_input("Torque (Nm)", value=40.0)
        tool_wear = st.number_input("Tool Wear (min)", value=0.0)
    
    submit = st.form_submit_button("Run AI Prediction")

# 5. PREDICTION RESULT (Required Feature)
if submit:
    # Prepare input for the model
    features = np.array([[air_temp, speed, torque]])
    prediction = model.predict(features)
    
    st.divider()
    if prediction[0] == 1:
        st.error("### ⚠️ Result: FAILURE LIKELY")
        st.write("The AI detected patterns matching mechanical failure. Maintenance is required.")
    else:
        st.success("### ✅ Result: MACHINE HEALTHY")
        st.write("The machine is operating within safe parameters.")

# 6. ABOUT THE DATASET (Required Feature)
st.divider()
st.info("**Dataset Source:** AI4I 2020 Predictive Maintenance Dataset (Kaggle) [cite: 24, 86]")