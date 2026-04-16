# House Price Prediction باستخدام Multiple Linear Regression

## Project Overview
This project demonstrates how to build and evaluate a **Multiple Linear Regression model** to predict house prices based on key property features.

The dataset is synthetically generated and includes:
- Area in square feet
- Number of bedrooms
- Age of the property

---

## Features Used
- `area_sqft` → Size of the house  
- `num_bedrooms` → Number of bedrooms  
- `age_years` → Age of the property  

### Target Variable
- `price_lakhs` → House price in lakhs (INR)

---

## Tasks Performed

### Task 1: Data Generation & Model Training
- Created a synthetic dataset with 60 records
- Built a Multiple Linear Regression model using `scikit-learn`
- Extracted:
  - Intercept
  - Feature coefficients
- Compared actual vs predicted values

---

### Task 2: Model Evaluation
Evaluated the model using:

- **MAE (Mean Absolute Error)**  
  → Measures average prediction error  

- **RMSE (Root Mean Squared Error)**  
  → Penalizes large errors more heavily  

- **R² Score**  
  → Indicates how well the model explains variance  

---

### Task 3: Residual Analysis
- Computed residuals (Actual - Predicted)
- Plotted histogram using Matplotlib
- Interpreted distribution to assess model performance

---

## Key Insights
- Price increases with area and number of bedrooms  
- Price decreases as property age increases  
- Residuals are approximately normally distributed → good model fit  

---

## Technologies Used
- Python  
- NumPy  
- Pandas  
- Scikit-learn  
- Matplotlib  

---

## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/ml-leakage-pipeline-muthu.git