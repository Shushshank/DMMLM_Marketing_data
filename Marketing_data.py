import streamlit as st
import pandas as pd
import pickle

# Load the trained Random Forest model
model = pickle.load(open('rf_model.pkl', 'rb'))

# Create a title for your app
st.title("Random Forest Model Prediction App")

# Create input fields for the features
# Replace 'feature1', 'feature2', etc. with the actual names of your features
feature1 = st.number_input("Feature 1")
feature2 = st.number_input("Feature 2")
# Add more input fields as needed

# Create a button to trigger the prediction
if st.button("Predict"):
  # Create a DataFrame with the user input
  input_data = pd.DataFrame({
      'Feature1': [feature1],
      'Feature2': [feature2],
      # Add more features as needed
  })

  # Make a prediction using the loaded model
  prediction = model.predict(input_data)

  # Display the prediction
  st.write("Prediction:", prediction[0])