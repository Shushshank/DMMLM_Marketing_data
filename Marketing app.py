import streamlit as st
import pandas as pd
import pickle

# Load the trained model
model = pickle.load(open('rf_model.pkl', 'rb'))

# Create a title for your app
st.title("Insurance Response Prediction App")

# Create input fields for the features
st.header("Enter Customer Information:")
age = st.number_input("Age", min_value=18, max_value=100, value=30)
driving_license = st.selectbox("Driving License", [0, 1])
region_code = st.number_input("Region Code", min_value=0, max_value=100, value=10)
previously_insured = st.selectbox("Previously Insured", [0, 1])
vehicle_age = st.selectbox("Vehicle Age", ["< 1 Year", "1-2 Year", "> 2 Years"])
vehicle_damage = st.selectbox("Vehicle Damage", ["Yes", "No"])
annual_premium = st.number_input("Annual Premium", min_value=0, value=1000)
policy_sales_channel = st.number_input("Policy Sales Channel", min_value=1, value=10)
vintage = st.number_input("Vintage", min_value=0, value=100)

# Create a button to make predictions
if st.button("Predict Response"):
    # Create a DataFrame with the input values
    input_data = pd.DataFrame({
        'Age': [age],
        'Driving_License': [driving_license],
        'Region_Code': [region_code],
        'Previously_Insured': [previously_insured],
        'Vehicle_Age': [vehicle_age],
        'Vehicle_Damage': [vehicle_damage],
        'Annual_Premium': [annual_premium],
        'Policy_Sales_Channel': [policy_sales_channel],
        'Vintage': [vintage]
    })

    # Make a prediction using the loaded model
    prediction = model.predict(input_data)[0]

    # Display the prediction
    if prediction == 1:
        st.success("The customer is likely to respond to the insurance offer.")
    else:
        st.warning("The customer is likely not to respond to the insurance offer.")

