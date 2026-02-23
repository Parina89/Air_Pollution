import streamlit as st
import pandas as pd
# import plotly.express as px
from sklearn.ensemble 
import RandomForestRegressor
from sklearn.model_selection 
import train_test_split

# 1. Page Config & Title
st.set_page_config(page_title="Air Pollution Detection", layout="wide")
st.title("🌍 Air Pollution Level Detection System")

# 2. Load and Prepare Data
@st.cache_data
def load_data():
    df = pd.read_csv('Data.csv')
    return df

df = load_data()

# 3. Sidebar - Input Features for Prediction
st.sidebar.header("Real-time Prediction Input")
pm25 = st.sidebar.slider("PM2.5 Level (µg/m³)", 0, 500, 50)
pm10 = st.sidebar.slider("PM10 Level (µg/m³)", 0, 500, 80)
no2 = st.sidebar.slider("NO2 Level (µg/m³)", 0, 200, 30)
co = st.sidebar.slider("CO Level (mg/m³)", 0.0, 10.0, 1.5)

# 4. Model Training
X = df[['PM2.5', 'PM10', 'NO2', 'CO']]
y = df['AQI']
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# 5. Prediction Logic
prediction = model.predict([[pm25, pm10, no2, co]])[0]

# Display Results
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Current Prediction")
    st.metric(label="Predicted AQI", value=f"{prediction:.2f}")
    
    # Categorize AQI
    if prediction <= 50:
        st.success("Air Quality: Good")
    elif prediction <= 100:
        st.warning("Air Quality: Moderate")
    else:
        st.error("Air Quality: Poor/Hazardous")

with col2:
    st.subheader("Pollutant Trends")
    fig = px.scatter(df, x="PM2.5", y="AQI", color="AQI", title="PM2.5 vs Predicted AQI")
    st.plotly_chart(fig, use_container_width=True)

# Data Overview
if st.checkbox("Show Raw Dataset"):
    st.write(df.head(10))
