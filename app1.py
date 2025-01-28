import streamlit as st
import joblib
import pandas as pd
import numpy as np
# Load the trained model
model = joblib.load('best_svm_model.pkl')
# Step 2: Prediction function 
def predict(features):
    """
    Predict class using the SVM model.
    :param features: List of feature values (e.g., [4.5, 2.3, 1.1])
    :return: Predicted class
    """
    return model.predict([features])[0]

# Step 3: Streamlit 
st.title("SVM Prediction App")
st.write("Enter the feature values for prediction:")

# Input fields
feature1 = st.number_input("Gender:")
feature2 = st.number_input("Age:")
feature3 = st.number_input("EstimatedSalary:")

# Predict button
if st.button("Predict"):
    # Make a prediction
    prediction = predict([feature1, feature2, feature3])
    st.success(f"The predicted class is: {prediction}")