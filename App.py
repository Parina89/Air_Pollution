import streamlit as st
import pickle
import numpy as np

st.set_page_config(page_title="Air Pollution Level Detection")

st.title("🌫 Air Pollution Level Detection System")
st.write("Enter pollutant values to predict AQI category")

# ✅ LOAD MODEL AT TOP LEVEL
try:
    model, imputer = pickle.load(open("model.pkl", "rb"))
except Exception as e:
    st.error("❌ Model not loaded. Please retrain and upload model.pkl")
    st.stop()

# Input fields
pm25 = st.number_input("PM2.5", 0.0, 500.0, 35.0)
pm10 = st.number_input("PM10", 0.0, 600.0, 80.0)
no2 = st.number_input("NO2", 0.0, 200.0, 20.0)
co = st.number_input("CO", 0.0, 10.0, 0.5)
o3 = st.number_input("O3", 0.0, 200.0, 30.0)

# Prediction button
if st.button("Predict AQI Level"):

    input_data = np.array([[pm25, pm10, no2, co, o3]])

    # Apply imputer
    input_data = imputer.transform(input_data)

    # Predict
    prediction = model.predict(input_data)

    st.subheader("Predicted AQI Category:")
    st.success(prediction[0])
