import streamlit as st
import pandas as pd
import pickle
from sklearn.ensemble import RandomForestClassifier

# Load the trained model
filename = 'rf_model.pkl'
rf_model = pickle.load(open(filename, 'rb'))

# Load the dataset (replace with your actual dataset path)
df = pd.read_csv('your_file.csv')  # Replace 'your_file.csv' with the actual file name

# Create the Streamlit app
st.title("Random Forest Model Prediction App")

# Display the dataset (optional)
st.subheader("Dataset Preview")
st.dataframe(df.head())

# Create input fields for user to enter data
st.subheader("Enter Input Features:")

# Get the feature names from your dataset
feature_names = df.columns[:-1]  # Assuming the last column is the target variable

input_features = {}
for feature in feature_names:
  input_features[feature] = st.number_input(f"Enter {feature}:")

# Create a button to make predictions
if st.button("Predict"):
  # Create a DataFrame from the user input
  input_df = pd.DataFrame([input_features])

  # Make prediction using the loaded model
  prediction = rf_model.predict(input_df)[0]

  # Display the prediction
  st.subheader("Prediction:")
  st.write(f"The predicted response is: {prediction}")

# You can add more features to the app, such as:
# - Displaying model performance metrics
# - Allowing users to upload their own data
# - Visualizing the data using charts and graphs
