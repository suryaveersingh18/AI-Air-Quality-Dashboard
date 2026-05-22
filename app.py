import streamlit as st
import joblib
import numpy as np
import requests
import plotly.express as px
import pandas as pd
from streamlit_js_eval import streamlit_js_eval

# ==========================================
# PAGE CONFIG
# ==========================================

st.set_page_config(
    page_title="AI Air Quality Dashboard",
    page_icon="🌍",
    layout="wide"
)

# ==========================================
# CUSTOM CSS
# ==========================================

st.markdown("""
<style>

body {
    background-color: #0e1117;
    color: white;
}

.main {
    background-color: #0e1117;
}

h1, h2, h3, h4, h5 {
    color: white;
}

.stMetric {
    background-color: #1c1f26;
    padding: 15px;
    border-radius: 12px;
}

.block-container {
    padding-top: 2rem;
}

.stButton > button {
    width: 100%;
    height: 3em;
    border-radius: 10px;
    background-color: #00ADB5;
    color: white;
    font-size: 18px;
    border: none;
}

</style>
""", unsafe_allow_html=True)

# ==========================================
# LOAD MODEL
# ==========================================

model = joblib.load("models/random_forest_aqi_model.pkl")

# ==========================================
# WEATHER API KEY
# ==========================================

API_KEY = "167d8e021677574f771dcdb20b84caf9"

# ==========================================
# TITLE
# ==========================================

st.title("🌍 AI Air Quality Monitoring Dashboard")

st.write(
    "Real-Time Air Pollution Prediction using Machine Learning"
)

# ==========================================
# LIVE GPS LOCATION
# ==========================================

st.sidebar.header("🌍 Live Location Detection")

st.sidebar.info(
    "Allow browser location access for automatic city detection."
)

# ==========================================
# GET BROWSER LOCATION
# ==========================================

location = streamlit_js_eval(
    js_expressions="""
    new Promise((resolve, reject) => {
        navigator.geolocation.getCurrentPosition(
            (position) => {
                resolve({
                    latitude: position.coords.latitude,
                    longitude: position.coords.longitude
                });
            },
            (error) => {
                reject(error);
            }
        );
    });
    """,
    key="get_location"
)

# ==========================================
# DEFAULT VALUES
# ==========================================

city = "Delhi"
latitude = None
longitude = None

# ==========================================
# GET CITY FROM COORDINATES
# ==========================================

if location is not None:

    try:

        latitude = location["latitude"]
        longitude = location["longitude"]

        st.sidebar.success(
            f"Latitude: {latitude:.4f}"
        )

        st.sidebar.success(
            f"Longitude: {longitude:.4f}"
        )

        # ==================================
        # REVERSE GEOCODING API
        # ==================================

        geo_url = f"""
        http://api.openweathermap.org/geo/1.0/reverse?lat={latitude}&lon={longitude}&limit=1&appid={API_KEY}
        """

        geo_response = requests.get(geo_url)

        geo_data = geo_response.json()

        if len(geo_data) > 0:

            city = geo_data[0]["name"]

    except:

        city = "Delhi"

# ==========================================
# MANUAL CITY INPUT
# ==========================================

city = st.sidebar.text_input(
    "Detected City",
    value=city
)

st.sidebar.success(f"Current City: {city}")

# ==========================================
# WEATHER API URL
# ==========================================

url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

# ==========================================
# WEATHER FETCH
# ==========================================

try:

    response = requests.get(url)

    weather_data = response.json()

    if response.status_code == 200:

        temperature = weather_data["main"]["temp"]
        humidity = weather_data["main"]["humidity"]
        weather = weather_data["weather"][0]["description"]

    else:

        temperature = 32
        humidity = 45
        weather = "Unavailable"

except:

    temperature = 32
    humidity = 45
    weather = "Unavailable"

# ==========================================
# WEATHER METRICS
# ==========================================

col1, col2, col3 = st.columns(3)

col1.metric(
    "🌡 Temperature",
    f"{temperature} °C"
)

col2.metric(
    "💧 Humidity",
    f"{humidity}%"
)

col3.metric(
    "☁ Weather",
    weather.title()
)

# ==========================================
# INPUT SECTION
# ==========================================

st.write("---")

st.subheader("Enter Pollutant Values")

col1, col2, col3 = st.columns(3)

with col1:

    co = st.number_input(
        "CO",
        min_value=0.0,
        value=20.0
    )

with col2:

    nh3 = st.number_input(
        "NH3",
        min_value=0.0,
        value=10.0
    )

with col3:

    no2 = st.number_input(
        "NO2",
        min_value=0.0,
        value=40.0
    )

col4, col5, col6 = st.columns(3)

with col4:

    ozone = st.number_input(
        "OZONE",
        min_value=0.0,
        value=80.0
    )

with col5:

    pm10 = st.number_input(
        "PM10",
        min_value=0.0,
        value=120.0
    )

with col6:

    so2 = st.number_input(
        "SO2",
        min_value=0.0,
        value=25.0
    )

# ==========================================
# AQI CATEGORY FUNCTION
# ==========================================

def get_category(pm25):

    if pm25 <= 30:
        return "🟢 Good"

    elif pm25 <= 60:
        return "🟡 Moderate"

    elif pm25 <= 90:
        return "🟠 Poor"

    elif pm25 <= 120:
        return "🔴 Unhealthy"

    else:
        return "🟣 Hazardous"

# ==========================================
# PREDICTION BUTTON
# ==========================================

if st.button("Predict Air Pollution"):

    input_data = np.array([
        [co, nh3, no2, ozone, pm10, so2]
    ])

    # ======================================
    # CONVERT TO DATAFRAME
    # ======================================

    input_df = pd.DataFrame(
        input_data,
        columns=[
            'CO',
            'NH3',
            'NO2',
            'OZONE',
            'PM10',
            'SO2'
        ]
    )

    # ======================================
    # MODEL PREDICTION
    # ======================================

    prediction = model.predict(input_df)

    predicted_pm25 = prediction[0]

    category = get_category(predicted_pm25)

    # ======================================
    # RESULTS
    # ======================================

    st.write("---")

    st.success(
        f"Predicted PM2.5 Level: {predicted_pm25:.2f}"
    )

    st.subheader(
        f"Air Quality Status: {category}"
    )

    # ======================================
    # PROGRESS BAR
    # ======================================

    progress_value = min(int(predicted_pm25), 100)

    st.progress(progress_value)

    # ======================================
    # METRIC CARDS
    # ======================================

    m1, m2, m3 = st.columns(3)

    m1.metric(
        "PM2.5 Value",
        f"{predicted_pm25:.2f}"
    )

    m2.metric(
        "City",
        city
    )

    m3.metric(
        "Temperature",
        f"{temperature} °C"
    )

    # ======================================
    # POLLUTANT ANALYSIS DATA
    # ======================================

    pollutant_data = pd.DataFrame({

        "Pollutants": [
            "CO",
            "NH3",
            "NO2",
            "OZONE",
            "PM10",
            "SO2"
        ],

        "Values": [
            co,
            nh3,
            no2,
            ozone,
            pm10,
            so2
        ]
    })

    # ======================================
    # BAR CHART
    # ======================================

    fig = px.bar(
        pollutant_data,
        x="Pollutants",
        y="Values",
        title="Pollutant Analysis",
        template="plotly_dark",
        text_auto=True
    )

    st.plotly_chart(
        fig,
        width='stretch'
    )

    # ======================================
    # PIE CHART
    # ======================================

    pie_fig = px.pie(
        pollutant_data,
        names="Pollutants",
        values="Values",
        title="Pollutant Contribution",
        template="plotly_dark"
    )

    st.plotly_chart(
        pie_fig,
        width='stretch'
    )

    # ======================================
    # HEALTH RECOMMENDATIONS
    # ======================================

    st.subheader("🏥 Health Recommendations")

    if predicted_pm25 <= 30:

        st.success(
            "Air quality is good and safe for outdoor activities."
        )

    elif predicted_pm25 <= 60:

        st.warning(
            "Sensitive people should avoid long outdoor exposure."
        )

    elif predicted_pm25 <= 90:

        st.warning(
            "Wear a mask and avoid heavy outdoor exercise."
        )

    else:

        st.error(
            "Avoid outdoor activities and stay indoors if possible."
        )

# ==========================================
# FOOTER
# ==========================================

st.write("---")

st.caption(
    "Developed using Streamlit, Machine Learning, Plotly, OpenWeather API, and Scikit-learn"
)