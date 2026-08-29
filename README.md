# Electricity Bill Forecaster

A beginner-friendly machine learning project that predicts a household/property's electricity cost (bill) using Multiple Linear Regression.

## Abstract

This project builds a supervised machine learning model to estimate electricity cost based on property characteristics such as site area, water consumption, recycling rate, utilisation rate, air quality index, issue resolution time, resident count, and structure type. The model is trained using Multiple Linear Regression and deployed through a simple web interface built with Streamlit.

## Introduction

Electricity billing is often only known after usage has already occurred. This project explores whether electricity cost can be reasonably predicted in advance using easily available property-level features, allowing for early estimation and planning.

## Problem Statement

Given various property and site characteristics, predict the electricity cost as accurately as possible using a simple, interpretable machine learning model.

## Objectives

- Understand and apply Multiple Linear Regression to a real-world dataset
- Perform data exploration, cleaning, and preprocessing
- Train and evaluate a regression model using standard metrics
- Build a simple interface to make predictions on new, user-provided data
- Save and reuse the trained model without retraining

## Dataset Description

- **Source:** Kaggle - Electricity Cost Prediction Dataset
- **Size:** 10,000 rows, 9 columns
- **Type:** Real-world dataset (not synthetic)

**Features (X):**
| Column | Description |
|---|---|
| site area | Size of the property/site |
| structure type | Category of building (Residential, Mixed-use, Industrial, Commercial) |
| water consumption | Water used |
| recycling rate | Recycling percentage |
| utilisation rate | Utilisation percentage of the site |
| air qality index | Air quality index |
| issue reolution time | Time taken to resolve issues |
| resident count | Number of residents |

**Target (y):**
| Column | Description |
|---|---|
| electricity cost | Actual electricity bill/cost for the property |

No missing values were found in the dataset.

## Technologies Used

- Python
- Pandas, NumPy
- Matplotlib
- Scikit-learn
- Streamlit (for the UI)
- Joblib (for saving the model)
- VS Code

## Methodology

1. Loaded and explored the dataset (shape, columns, missing values, statistics)
2. Preprocessed the data: converted the categorical `structure type` column into numeric form using one-hot encoding
3. Visualised relationships between features and electricity cost using scatter plots
4. Split the data into training (80%) and testing (20%) sets
5. Trained a Multiple Linear Regression model on the training set
6. Evaluated the model on the test set using MAE, MSE, RMSE, and R² score
7. Built a script to accept manual property details and predict the electricity bill
8. Saved the trained model using Joblib for reuse without retraining
9. Built a simple Streamlit web UI for interactive predictions

## Multiple Linear Regression

Multiple Linear Regression models the relationship between several input features and one output as:

```
y = b0 + b1*x1 + b2*x2 + ... + bn*xn
```

Here, `y` is the predicted electricity cost, and each `b` coefficient represents how strongly the corresponding feature influences the cost.

## Results

| Metric | Value |
|---|---|
| MAE | 245.65 |
| MSE | 97381.90 |
| RMSE | 312.06 |
| R² Score | 0.922 |

The model explains approximately 92% of the variance in electricity cost, indicating a strong linear relationship between the chosen features and the target.

## Conclusion

Multiple Linear Regression proved to be a suitable and interpretable model for this dataset, achieving a high R² score without requiring complex algorithms. The project demonstrates a complete, practical ML workflow from raw data to a usable prediction interface.

## Future Scope

- Experiment with feature engineering to further reduce error
- Compare performance against other regression techniques
- Add input validation and error handling to the UI
- Deploy the Streamlit app online for public access

## How to Run the Project

1. Clone the repository:
   ```
   git clone https://github.com/Aleenamehreen04/electricity-bill-forecaster.git
   cd electricity-bill-forecaster
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

3. Run the model training/evaluation scripts (optional, model is already saved):
   ```
   python src/evaluate_model.py
   ```

4. Launch the web app:
   ```
   streamlit run app.py
   ```

5. Enter property details in the browser form to get a predicted electricity bill.

**Note:** The Streamlit app (`app.py`) is not deployed online, so it cannot be run directly from this repository link. A screen recording demonstrating the app running locally has been provided separately to show its functionality.