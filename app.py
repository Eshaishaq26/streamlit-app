import joblib
import streamlit as st
import pandas as pd
import numpy as np
# Load the trained model
model = joblib.load('best_svm_model.pkl')
# Title of the application
st.title('SVM Model Predictor')
# Collect user inputs
input_feature1 = st.number_input('Input Feature 1')
input_feature2 = st.number_input('Input Feature 2')
# Add more input fields as required
# Create a DataFrame for prediction
input_data = pd.DataFrame({
    'feature1': [input_feature1],
    'feature2': [input_feature2]
    # Add more features as required
})
# Prediction button
if st.button('Predict'):
    prediction = model.predict(input_data)
    st.write(f'Prediction: {prediction[0]}')


