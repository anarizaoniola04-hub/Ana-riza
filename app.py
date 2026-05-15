%%writefile app.py
import streamlit as st
import numpy as np
from sklearn.tree import DecisionTreeClassifier

# Web App Title & Program Alignment [cite: 9, 82]
st.title("Machine Failure Prediction System")
st.write("Program: Mechanical Engineering")

# Simple AI Model Logic 
X = np.array([[300, 1500, 40], [310, 1200, 60], [290, 1600, 30], [320, 1000, 70]])
y = np.array([0, 1, 0, 1]) # 0=Healthy, 1=Failure
model = DecisionTreeClassifier().fit(X, y)

# Input Form [cite: 78]
with st.form("sensor_input"):
    temp = st.number_input("Air Temperature (K)")
    speed = st.number_input("Rotational Speed (RPM)")
    torque = st.number_input("Torque (Nm)")
    submit = st.form_submit_button("Predict")

# Result Display [cite: 79, 80]
if submit:
    prediction = model.predict([[temp, speed, torque]])
    if prediction[0] == 1:
        st.error("Result: Potential Failure Detected")
    else:
        st.success("Result: Machine Operating Normally")