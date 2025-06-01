import tensorflow as tf
import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained ANN model
model = tf.keras.models.load_model('model.h5')

# Load the scaler
with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# Streamlit app title
st.title('Heart Attack Prediction')

# User Inputs (user-friendly labels)
age = st.slider('Age', 18, 92)
sex = st.selectbox('Gender (0 = Female, 1 = Male)', [0, 1])
cp = st.slider('Chest Pain Type (cp)', 0, 6)
trestbps = st.number_input('Resting Blood Pressure (trestbps)', step=1)
fbs = st.selectbox('Fasting Blood Sugar > 120 mg/dl (fbs)', [0, 1])
restecg = st.slider('Resting Electrocardiographic Results (restecg)', 0, 4)
thalach = st.number_input('Maximum Heart Rate Achieved (thalach)', step=1)
exang = st.selectbox('Exercise Induced Angina (exang)', [0, 1])
oldpeak = st.number_input('Oldpeak (ST depression)', step=0.1)
ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy (ca)', [0, 1, 2, 3, 4])

# Prepare the input data with correct feature names (must match training)
input_data = pd.DataFrame({
    'age': [age],
    'sex': [sex],
    'cp': [cp],
    'trestbps': [trestbps],
    'fbs': [fbs],
    'restecg': [restecg],
    'thalach': [thalach],
    'exang': [exang],
    'oldpeak': [oldpeak],
    'ca': [ca]
})

# Scale the input data
input_scaled = scaler.transform(input_data)

# Predict
predictions = model.predict(input_scaled)
predict_proba = predictions[0][0]

# Show result
if predict_proba > 0.5:
    st.error(" The patient is likely to have a heart attack.")
else:
    st.success(" The patient is not likely to have a heart attack.")
