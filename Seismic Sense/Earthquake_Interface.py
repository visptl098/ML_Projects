import streamlit as st
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timezone

# Load the model from the file
model_filename = 'svr_magnitude_model.pkl'
with open(model_filename, 'rb') as file:
    svr_magnitude_model = pickle.load(file)

# Define the function to convert date and time to a Unix timestamp
def convert_to_timestamp(date, time):
    dt = datetime.strptime(date + ' ' + time, '%m/%d/%Y %H:%M:%S')
    # Manually calculate the timestamp
    timestamp = int(dt.replace(tzinfo=timezone.utc).timestamp())
    return timestamp

# Streamlit user interface
st.title("Earthquake Magnitude Prediction")

st.write("Enter the date, time, latitude, and longitude to predict the earthquake magnitude:")

date = st.text_input("Date (MM/DD/YYYY)", "01/01/2024")
time = st.text_input("Time (HH:MM:SS)", "12:00:00")
latitude = st.number_input("Latitude", value=0.0)
longitude = st.number_input("Longitude", value=0.0)

if st.button("Predict"):
    try:
        # Convert date and time to timestamp
        timestamp = convert_to_timestamp(date, time)
        
        # Prepare the input data for the model
        input_data = np.array([[timestamp, latitude, longitude]])
        
        # Predict the magnitude using the loaded model
        predicted_magnitude = svr_magnitude_model.predict(input_data)
        
        # Display the prediction
        st.write(f"Predicted Earthquake Magnitude: {predicted_magnitude[0]:.2f}")
    except Exception as e:
        st.write(f"An error occurred: {e}")

