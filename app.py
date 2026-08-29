# ==============================================================
# Electricity Bill Forecaster - Simple Web UI (Streamlit)
# ==============================================================
# This app loads the already-trained model (saved in Stage 12)
# and lets the user enter property details through a simple web
# form to get a predicted electricity bill.
# ==============================================================

import streamlit as st
import pandas as pd
import joblib

# ------------------------------
# Step 1: Load the saved model
# ------------------------------
# We do NOT retrain here - the model was already trained and
# saved as a .pkl file in Stage 12, so we just load it.
model = joblib.load("model/electricity_model.pkl")

# ------------------------------
# Step 2: Define the expected column order
# ------------------------------
# This must exactly match the column order the model was
# trained on (from X.columns during training).
COLUMN_ORDER = [
    "site area",
    "water consumption",
    "recycling rate",
    "utilisation rate",
    "air qality index",
    "issue reolution time",
    "resident count",
    "structure type_Industrial",
    "structure type_Mixed-use",
    "structure type_Residential",
]

# ------------------------------
# Step 3: Build the web page
# ------------------------------
st.title("Electricity Bill Forecaster")
st.write("Enter property details below to predict the estimated electricity bill.")

# Input widgets - each one collects one feature value
site_area = st.number_input("Site Area", min_value=0.0, value=2000.0)
water_consumption = st.number_input("Water Consumption", min_value=0.0, value=3000.0)
recycling_rate = st.number_input("Recycling Rate", min_value=0.0, max_value=100.0, value=50.0)
utilisation_rate = st.number_input("Utilisation Rate", min_value=0.0, max_value=100.0, value=60.0)
air_quality_index = st.number_input("Air Quality Index", min_value=0.0, value=150.0)
issue_resolution_time = st.number_input("Issue Resolution Time", min_value=0.0, value=20.0)
resident_count = st.number_input("Resident Count", min_value=0.0, value=100.0)
structure_type = st.selectbox(
    "Structure Type",
    ["Commercial", "Industrial", "Mixed-use", "Residential"]
)

# ------------------------------
# Step 4: Predict when button is clicked
# ------------------------------
if st.button("Predict Electricity Bill"):

    # Convert structure type into the same one-hot format as training
    industrial = 1 if structure_type == "Industrial" else 0
    mixed_use = 1 if structure_type == "Mixed-use" else 0
    residential = 1 if structure_type == "Residential" else 0

    # Arrange input in the exact column order the model expects
    input_data = pd.DataFrame([[
        site_area,
        water_consumption,
        recycling_rate,
        utilisation_rate,
        air_quality_index,
        issue_resolution_time,
        resident_count,
        industrial,
        mixed_use,
        residential
    ]], columns=COLUMN_ORDER)

    # Get prediction and display it
    predicted_cost = model.predict(input_data)
    st.success(f"Predicted Electricity Bill: {round(predicted_cost[0], 2)}")