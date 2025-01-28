import joblib
import streamlit as st
import pandas as pd
import numpy as np

# Load the trained model
model = joblib.load('best_svm_model.pkl')

# Title of the application
st.title('SVM Model Predictor')

# Collect user inputs with descriptive labels
gender = st.selectbox('Select Gender:', ['Male', 'Female'])
age = st.number_input('Enter Age:', min_value=0, max_value=100, step=1)
# Remove the third input field for Estimated Salary

# Convert Gender to numerical value for the model
gender_value = 1 if gender == 'Male' else 0

# Create a DataFrame for prediction
input_data = pd.DataFrame({
    'Gender': [gender_value],
    'Age': [age]
})

# Ensure the input DataFrame matches the expected format
st.write('Input Data:')
st.write(input_data)

# Prediction button
if st.button('Predict'):
    try:
        prediction = model.predict(input_data)
        st.write(f'Prediction: {prediction[0]}')
        # Optionally, interpret the prediction (e.g., 0 = Not Purchased, 1 = Purchased)
        result = 'Purchased' if prediction[0] == 1 else 'Not Purchased'
        st.write(f'The model predicts that the purchase status is: {result}')
    except ValueError as e:
        st.error(f"Error making prediction: {e}")
