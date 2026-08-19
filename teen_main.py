import streamlit as st 
import numpy as np
import pandas as pd 
import joblib as jb 

# 1. Set Page Configuration
st.set_page_config(
    page_title="Digital Well-Being Predictor", 
    page_icon="fear (1).png", 
    layout="wide"
)   

# 2. Introduction Section 
st.markdown("=" * 120)
st.image("fear.png", width=150) # Added width control for a cleaner layout look
st.title("Welcome to the Digital Well-Being Predictor")
st.write("This application leverages machine learning to assess and forecast a student's digital mental well-being based on their social media usage patterns, helping educators and guardians promote healthier online habits.")
st.markdown("=" * 120)

# 3. Model Loading (Cached)
@st.cache_resource 
def load_model():
    try: 
        algorithm = jb.load("Teens_mental_model.pki")
        return algorithm
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None
        
Model = load_model()

# 4. Input Features Section
st.subheader("== Input Features Values to Predict Digital Mental Well-Being ==")

# Split page segment into two columns
col1, col2 = st.columns(2)

with col1:
    age = col1.number_input("Age of Student (13 - 19)", min_value=13, max_value=19, step=1)
    digital_social_hr = col1.number_input("Daily Social Media Hours (1 - 8)", min_value=1.0, max_value=8.0, step=1.0)
    Platform = col1.selectbox("Platform Usage", ["Facebook", "TikTok", "Instagram", "YouTube", "All Platforms"])
    physical_activity = col1.number_input("Physical Activity", min_value=0.0, max_value=2.0, step=0.05)
    sleep_hours = col1.number_input("Sleep Hours", min_value=4.0, max_value=9.0, step=1.0)
    screen_time_before_bed = col1.number_input("Screen Time before Bed", min_value=0.5, max_value=3.0, step=1.0)
    social_interaction_level = col1.selectbox("Social Interaction Level", ['low', 'medium', 'high'])

with col2:
    anxiety_level = col2.number_input("Anxiety level (1 - 10)", min_value=1, max_value=10, step=1)
    stress_level = col2.number_input("Stress level (1 - 10)", min_value=1, max_value=10, step=1)
    addiction_level = col2.number_input("Addiction level (1 - 10)", min_value=1, max_value=10, step=1)
    depression_label = col2.number_input("Depression label (0 - 1)", min_value=0, max_value=1, step=1)
    mental_health_risk_score = col2.number_input("Mental Health Risk Score (3 - 30)", min_value=3, max_value=30, step=1)
    sleep_quality = col2.selectbox("Sleep Quality", ['Fair', 'Good', 'Poor'])

# 5. Categorical Encoding Mappings
platform_index = {"Facebook": 0, 'TikTok': 1, "Instagram": 2, 'YouTube': 3, 'All Platforms': 4}
s_i_index = {'low': 0, 'medium': 2, 'high': 1}
s_q_index = {'Fair': 0, 'Good': 1, 'Poor': 2}

# Format inputs into a Pandas DataFrame ensuring feature names align with training expectations
inputs = pd.DataFrame({
    "age": [age], 
    "daily_social_media_hours": [digital_social_hr],
    "platform_usage": [platform_index.get(Platform, 0)],
    "sleep_hours": [sleep_hours], 
    "screen_time_before_sleep": [screen_time_before_bed],
    "physical_activity": [physical_activity],
    "social_interaction_level": [s_i_index.get(social_interaction_level, 0)],
    "stress_level": [stress_level],
    "anxiety_level": [anxiety_level],
    "addiction_level": [addiction_level],
    "depression_label": [depression_label],
    "mental_health_risk_score": [mental_health_risk_score],
    "sleep_quality": [s_q_index.get(sleep_quality, 0)]
})

# 6. Session State Initialization for Persistent Prediction Results
if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None

# 7. Prediction Logic & Action Button
if st.button("Predict", type="primary"):
    if Model is None:
        st.error("Prediction model is not loaded. Please check file path.")
    else:
        try:
            # Run the prediction
            prediction = Model.predict(inputs)
            st.session_state.prediction_result = prediction[0]
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")
            st.session_state.prediction_result = None

# 8. Display Persistent Results
if st.session_state.prediction_result is not None:
    st.markdown("---")
    st.subheader("Prediction Results")
    res = st.session_state.prediction_result
    
    if res == 0:
        st.error("The Student's Digital well-being is at risk.")
    elif res == 1:
        st.warning("The Student's Digital well-being is Moderate")
    elif res == 2:
        st.success("The Student's Digital well-being is Healthy")
    else:
        st.error("No valid prediction category was returned.")