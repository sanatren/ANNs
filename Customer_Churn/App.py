import streamlit as st
import tensorflow as tf
import numpy as np
import pandas as pd
import pickle
import requests
from streamlit_lottie import st_lottie
import json

# -----------------------
# Load Resources
# -----------------------
model = tf.keras.models.load_model('model.h5')

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)

# -----------------------
# Utility Functions
# -----------------------
def load_lottieurl(url: str):
    """Load a Lottie animation from a given URL."""
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception as e:
        st.warning(f"Error loading Lottie animation: {e}")
        return None

# Valid Lottie Animation URLs
lottie_confetti = load_lottieurl("https://assets10.lottiefiles.com/packages/lf20_pu9ukanc.json")
lottie_predict = load_lottieurl("https://assets4.lottiefiles.com/packages/lf20_h8chuh1v.json")
lottie_header = load_lottieurl("https://assets7.lottiefiles.com/packages/lf20_dcatp5cr.json")

# -----------------------
# Page Configuration
# -----------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600&display=swap');

    html, body, [class*="css"]  {
        font-family: 'Montserrat', sans-serif;
    }

    .stButton button {
        background: linear-gradient(135deg, #9C27B0, #E91E63);
        color: #FFFFFF;
        border-radius: 8px;
    }

    .stButton button:hover {
        background: linear-gradient(135deg, #E91E63, #9C27B0);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------
# Header Section
# -----------------------
col1, col2, col3 = st.columns([1, 3, 1])
with col2:
    if lottie_header:
        st_lottie(lottie_header, height=200, key="header_animation")
    else:
        st.markdown("### Customer Churn Prediction")

st.title("Customer Churn Prediction")
st.subheader("Predict the likelihood of a customer churning based on their profile.")

# -----------------------
# User Input Section
# -----------------------
st.write("**Please provide the customer's details below:**")

col_left, col_right = st.columns(2)
with col_left:
    geography = st.selectbox('Geography', onehot_encoder_geo.categories_[0])
    gender = st.selectbox('Gender', label_encoder_gender.classes_)
    age = st.slider('Age', 18, 92, value=30)
    credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
    tenure = st.slider('Tenure (Years)', 0, 10, value=5)

with col_right:
    balance = st.number_input('Balance', min_value=0.0, value=0.0)
    estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)
    num_of_products = st.slider('Number of Products', 1, 4, value=1)
    has_cr_card = st.selectbox('Has Credit Card?', [0, 1], index=1)
    is_active_member = st.selectbox('Is Active Member?', [0, 1], index=1)

# Prepare input data
input_data = {
    'CreditScore': credit_score,
    'Gender': label_encoder_gender.transform([gender])[0],
    'Age': age,
    'Tenure': tenure,
    'Balance': balance,
    'NumOfProducts': num_of_products,
    'HasCrCard': has_cr_card,
    'IsActiveMember': is_active_member,
    'EstimatedSalary': estimated_salary
}

# Encode geography
geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

# Combine all inputs
input_df = pd.DataFrame([input_data])
input_combined = pd.concat([input_df, geo_encoded_df], axis=1)

# Scale input data
input_scaled = scaler.transform(input_combined)

# -----------------------
# Prediction Section
# -----------------------
st.write("**Click the button below to predict churn probability:**")

if lottie_predict:
    st_lottie(lottie_predict, height=150, key="predict_animation")

if st.button('Predict'):
    prediction = model.predict(input_scaled)
    churn_probability = prediction[0][0]

    st.markdown(f"### Churn Probability: **{churn_probability:.2f}**")

    if churn_probability > 0.5:
        st.error('The customer is likely to churn.')
        if lottie_confetti:
            st_lottie(lottie_confetti, height=200, key="confetti_animation")
    else:
        st.success('The customer is not likely to churn.')
        st.balloons()
