import streamlit as st
import pickle
import numpy as np
import pandas as pd
import os
# Load model
model.load(open("model (1).pkl"))

# # Load model safely
# if os.path.exists("model (1).pkl"):
#     with open("model (1).pkl", "rb") as f:
#         model = pickle.load(f)
# else:
#     st.error("Model file not found! Please run train_model.py first.")
#     st.stop()

st.set_page_config(page_title="Air Pollution Level Detection", layout="centered")

st.title(" Air Pollution Level Detection System")
st.write("Enter pollutant values to predict AQI category")

# Input fields
pm25 = st.number_input("PM2.5")
pm10 = st.number_input("PM10")
no2 = st.number_input("NO2")
co = st.number_input("CO")
o3 = st.number_input("O3")

if st.button("Predict AQI Level"):

    input_data = np.array([[pm25, pm10, no2, co, o3]])  # lowercase variables
    prediction = model.predict(input_data)
    
    st.subheader("Predicted Air Quality Level:")
    st.success(prediction[0])


    # Color indicator
    if prediction[0] == "Good":
        st.markdown("Air quality is satisfactory")
    elif prediction[0] == "Moderate":
        st.markdown("Air quality is acceptable")
    elif prediction[0] == "Unhealthy":
        st.markdown("Sensitive groups may experience effects")
    elif prediction[0] == "Very Unhealthy":
        st.markdown("Health alert for everyone")
    else:
        st.markdown("Hazardous air quality")

# Show dataset
st.subheader("📊 Sample Air Quality Dataset")
data = pd.read_csv("Data.csv")
st.dataframe(data)

# Optional: Plot AQI distribution
st.subheader("📈 AQI Category Distribution")
st.bar_chart(data["AQI_Category"].value_counts())
