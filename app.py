import streamlit as st
import pandas as pd
import numpy as np
import joblib


# Load the trained model
model = joblib.load("sonar_final_model.pkl")


# Application title
st.title("Sonar Rock vs Mine Classification")

st.write(
    "Enter the 60 sonar signal measurements below "
    "to classify the object as Rock or Mine."
)


# Input fields for 60 sonar features
features = []

for i in range(60):
    value = st.number_input(
        f"Sonar Feature {i + 1}",
        value=0.0,
        format="%.4f"
    )
    features.append(value)


# Prediction
if st.button("Predict"):

    # Convert input into model format
    new_sample = np.array(features).reshape(1, -1)

    # Make prediction
    prediction = model.predict(new_sample)

    # Get prediction probabilities
    probability = model.predict_proba(new_sample)

    # Convert numerical prediction to label
    if prediction[0] == 0:
        result = "Rock"
    else:
        result = "Mine"

    # Display result
    st.subheader("Prediction Result")

    st.success(f"The Predicted Object is a {result}")

    # Display probabilities
    st.write(
        f"Probability of Rock: "
        f"{probability[0][0] * 100:.2f}%"
    )

    st.write(
        f"Probability of Mine: "
        f"{probability[0][1] * 100:.2f}%"
    )
