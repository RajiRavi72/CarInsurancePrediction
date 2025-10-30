import streamlit as st
import pandas as pd
import joblib
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

# ------------------- PAGE SETUP -------------------
st.set_page_config(
    page_title="Car Insurance Claim Prediction",
    page_icon="🚗",
    layout="wide"
)

# ------------------- LOAD MODEL -------------------
st.title("🚗 Car Insurance Claim Prediction App")

st.markdown("""
This app predicts whether a customer will make a **car insurance claim**  
in the next policy period based on demographic, vehicle, and policy information.
""")

try:
    model = joblib.load("lgbm_best_model.pkl")
    preprocessor = joblib.load("preprocessor.pkl")
    st.success("✅ Model and Preprocessor Loaded Successfully")
except Exception as e:
    st.error(f"❌ Error loading model or preprocessor: {e}")
    st.stop()

st.divider()

# ------------------- INPUT FORM -------------------
st.header("Enter Customer Details")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        policy_tenure = st.number_input("Policy Tenure (years)", min_value=0.0, max_value=10.0, step=0.1)
        age_of_car = st.number_input("Age of Car (normalized)", min_value=0.0, max_value=1.0, step=0.01)
        age_of_policyholder = st.number_input("Age of Policyholder (normalized)", min_value=0.0, max_value=1.0, step=0.01)
        area_cluster = st.selectbox("Area Cluster", ["C1", "C2", "C3", "C4", "C5"])
    
    with col2:
        make = st.selectbox("Make", ["A", "B", "C", "D"])
        segment = st.selectbox("Segment", ["A", "B", "C"])
        fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
        transmission_type = st.selectbox("Transmission Type", ["Manual", "Automatic"])

    with col3:
        airbags = st.number_input("Airbags", min_value=0, max_value=10, step=1)
        ncap_rating = st.number_input("NCAP Rating", min_value=0, max_value=5, step=1)
        is_esc = st.selectbox("ESC Available?", ["Yes", "No"])
        is_brake_assist = st.selectbox("Brake Assist?", ["Yes", "No"])

    submit = st.form_submit_button("🔍 Predict Claim Probability")

# ------------------- PREDICTION -------------------
if submit:
    try:
        # Prepare DataFrame for input
        input_data = pd.DataFrame([{
            "policy_tenure": policy_tenure,
            "age_of_car": age_of_car,
            "age_of_policyholder": age_of_policyholder,
            "area_cluster": area_cluster,
            "make": make,
            "segment": segment,
            "fuel_type": fuel_type,
            "transmission_type": transmission_type,
            "airbags": airbags,
            "ncap_rating": ncap_rating,
            "is_esc": is_esc,
            "is_brake_assist": is_brake_assist
        }])

        # --- Align columns with preprocessor ---
        expected_cols = preprocessor.feature_names_in_
        missing_cols = [col for col in expected_cols if col not in input_data.columns]

        for col in missing_cols:
            input_data[col] = None

        input_data = input_data[expected_cols]

        # --- Identify numeric & categorical columns from the preprocessor ---
        numeric_features, categorical_features = [], []
        if isinstance(preprocessor, ColumnTransformer):
            for name, transformer, cols in preprocessor.transformers_:
                if transformer is not None:
                    if isinstance(transformer, Pipeline):
                        for step_name, step_obj in transformer.steps:
                            if isinstance(step_obj, StandardScaler):
                                numeric_features.extend(cols)
                            elif isinstance(step_obj, OneHotEncoder):
                                categorical_features.extend(cols)
                    elif isinstance(transformer, StandardScaler):
                        numeric_features.extend(cols)
                    elif isinstance(transformer, OneHotEncoder):
                        categorical_features.extend(cols)

        # --- Fix Data Types ---
        for col in numeric_features:
            if col in input_data.columns:
                input_data[col] = pd.to_numeric(input_data[col], errors='coerce')

        for col in categorical_features:
            if col in input_data.columns:
                input_data[col] = input_data[col].astype(str)

        # --- Transform safely ---
        X_processed = preprocessor.transform(input_data)

        # --- Predict ---
        prob = model.predict_proba(X_processed)[0][1]
        pred = model.predict(X_processed)[0]

        # --- Display Results ---
        st.subheader("🧭 Prediction Result")
        st.metric(label="Claim Probability", value=f"{prob:.2%}")
        st.metric(label="Predicted Class", value="Claim" if pred == 1 else "No Claim")

        if prob > 0.5:
            st.error("⚠️ High Risk: Customer is **likely to make a claim.**")
        else:
            st.success("✅ Low Risk: Customer is **unlikely to make a claim.**")

    except Exception as e:
        st.error(f"⚠️ Prediction failed: {e}")
