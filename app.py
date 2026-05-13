%%writefile app.py
import streamlit as st
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import numpy as np

# 1. Project Overview (Required Web Feature) [cite: 77, 82]
st.title("Mechanical Engineering: Machine Failure Predictor")
st.write("This AI tool predicts if industrial equipment will fail based on sensor data.")

# 2. Simulated AI Model Training [cite: 50, 86]
def train_simple_model():
    # Features: Air Temp, Process Temp, Rotational Speed, Torque
    X = np.array([
        [298, 308, 1500, 40], [302, 312, 1350, 55], 
        [305, 315, 1200, 65], [295, 305, 1700, 30]
    ])
    y = np.array([0, 0, 1, 0]) # 0 = Healthy, 1 = Failure
    model = DecisionTreeClassifier()
    model.fit(X, y)
    return model

model = train_simple_model()

# 3. Input Form (Required Web Feature) [cite: 78]
st.header("Enter Sensor Readings")
with st.form("input_form"):
    air_temp = st.number_input("Air Temperature (K)", value=300)
    process_temp = st.number_input("Process Temperature (K)", value=310)
    speed = st.number_input("Rotational Speed (RPM)", value=1500)
    torque = st.number_input("Torque (Nm)", value=40)
    submit = st.form_submit_button("Predict Machine Health")

# 4. AI Prediction Result (Required Web Feature) [cite: 79, 80]
if submit:
    features = np.array([[air_temp, process_temp, speed, torque]])
    prediction = model.predict(features)
    if prediction[0] == 1:
        st.error("Result: FAILURE LIKELY. Schedule immediate maintenance.")
    else:
        st.success("Result: MACHINE HEALTHY. Operations can continue.")

# 5. About Page (Required Web Feature) [cite: 81]
st.divider()
st.subheader("About this Project")
st.write("Model: Decision Tree Classifier [cite: 52]")
st.write("Source: AI4I 2020 Predictive Maintenance Dataset [cite: 24]")