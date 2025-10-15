import streamlit as st
import pandas as pd
import numpy as np  
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor

# Load the trained model and scaler
data_dir = Path(__file__).parent / "data"
model_path = data_dir / "C:/Users/user/Documents/air_quality_forecast/rf_pm25_model.joblib"
scaler_path = data_dir / "C:/Users/user/Documents/air_quality_forecast/scaler_pm25_model.joblib"
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

print("Model and scaler loaded successfully.")

st.title("Air Quality Prediction App")
st.write("Predict PM2.5 levels based on environmental factors.")
st.image("C:/Users/user/Documents/air_quality_forecast/istockphoto.jpg", use_container_width=True)
st.write("This app uses a Random Forest model to predict PM2.5 levels based on various environmental inputs.")
st.write("Developed by Chuks Ugbome")
st.write("Data Source: [Open Africa Air Quality Dataset](https://open.africa/dataset/sensorsafrica-airquality-archive-lagos)")
st.write("Model: Random Forest Regressor")
st.write("Scaler: StandardScaler")
st.write("Date: October 2025")

# Function to get user inputs
def get_user_inputs():
    st.sidebar.header("Input Environmental Factors")
    PM1 = st.sidebar.slider("PM1 (µg/m³)", 0.0, 500.0, 20.0)
    PM10 = st.sidebar.slider("PM10 (µg/m³)", 0.0, 500.0, 50.0)
    humidity = st.sidebar.slider("Humidity (%)", 0.0, 100.0, 50.0)
    temperature = st.sidebar.slider("Temperature (°C)", -10.0, 50.0, 20.0)

    data = {
        "PM1": PM1,
        "PM10": PM10,
        "humidity": humidity,
        "temperature": temperature
    }
    features = pd.DataFrame(data, index=[0])
    return features

# Get user inputs
user_inputs = get_user_inputs()
st.subheader("User Input Parameters")
st.write(user_inputs)
# Scale the inputs
scaled_inputs = scaler.transform(user_inputs)
# Make prediction
prediction = model.predict(scaled_inputs)
st.subheader("Predicted PM2.5 Level (µg/m³)")
st.write(f"{prediction[0]:.2f}")
