# ==============================================================
# Electricity Bill Forecaster - Manual Prediction Script
# ==============================================================
# This script trains a Multiple Linear Regression model on the
# electricity cost dataset, then lets the user manually enter
# property details to get a predicted electricity bill.
# ==============================================================

import pandas as pd
from sklearn.linear_model import LinearRegression

# ------------------------------
# Step 1: Load the dataset
# ------------------------------
df = pd.read_csv("dataset/electricity_cost_dataset.csv")

# ------------------------------
# Step 2: Preprocess the data
# ------------------------------
# 'structure type' is a text column (e.g. Residential, Mixed-use).
# Linear Regression only works with numbers, so we convert this
# text column into multiple 0/1 columns using One-Hot Encoding.
# drop_first=True removes one category (Commercial) to avoid
# redundancy - if all other structure columns are 0, it means
# the property is Commercial.
df = pd.get_dummies(df, columns=["structure type"], drop_first=True)

# Separate features (X) and target (y)
# X = everything the model uses to predict
# y = what we are trying to predict (electricity cost)
X = df.drop("electricity cost", axis=1)
y = df["electricity cost"]

# ------------------------------
# Step 3: Train the model
# ------------------------------
# We train on the full dataset here since this script is for
# real predictions, not for measuring accuracy (that was already
# done separately in evaluate_model.py using a train-test split).
model = LinearRegression()
model.fit(X, y)
import joblib

joblib.dump(model, "model/electricity_model.pkl")
# ------------------------------
# Step 4: Take manual input from the user
# ------------------------------
print("Enter the property details to predict the electricity bill:\n")

site_area = float(input("Enter site area: "))
water_consumption = float(input("Enter water consumption: "))
recycling_rate = float(input("Enter recycling rate: "))
utilisation_rate = float(input("Enter utilisation rate: "))
air_quality_index = float(input("Enter air quality index: "))
issue_resolution_time = float(input("Enter issue resolution time: "))
resident_count = float(input("Enter resident count: "))
structure_type = input("Enter structure type (Mixed-use / Residential / Industrial / Commercial): ")

# ------------------------------
# Step 5: Convert structure type to match training format
# ------------------------------
# The model was trained with these one-hot columns:
#   structure type_Industrial
#   structure type_Mixed-use
#   structure type_Residential
# (Commercial has no column - it is the baseline category,
# represented by all three columns being 0)
industrial = 1 if structure_type == "Industrial" else 0
mixed_use = 1 if structure_type == "Mixed-use" else 0
residential = 1 if structure_type == "Residential" else 0

# ------------------------------
# Step 6: Arrange input in the same column order as training data
# ------------------------------
# The model expects the columns in the exact same order it was
# trained on, so we build a single-row DataFrame using X.columns
# to guarantee that order automatically.
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
]], columns=X.columns)

# ------------------------------
# Step 7: Predict and display result
# ------------------------------
predicted_cost = model.predict(input_data)
print("\nPredicted Electricity Cost/Bill:", round(predicted_cost[0], 2))