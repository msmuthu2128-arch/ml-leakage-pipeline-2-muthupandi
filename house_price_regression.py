#Task 1: Create Dataset & Train Model

# Import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Set seed for reproducibility
np.random.seed(42)

# Create synthetic dataset (50+ records)
n = 60
area_sqft = np.random.randint(600, 3000, n)
num_bedrooms = np.random.randint(1, 5, n)
age_years = np.random.randint(0, 30, n)

# Create target with some realistic relationship + noise
price_lakhs = (
    area_sqft * 0.05 +
    num_bedrooms * 10 -
    age_years * 0.8 +
    np.random.normal(0, 5, n)
)

# Create DataFrame
df = pd.DataFrame({
    'area_sqft': area_sqft,
    'num_bedrooms': num_bedrooms,
    'age_years': age_years,
    'price_lakhs': price_lakhs
})

# Features & target
X = df[['area_sqft', 'num_bedrooms', 'age_years']]
y = df['price_lakhs']

# Train model
model = LinearRegression()
model.fit(X, y)

# Predictions
y_pred = model.predict(X)

# Print intercept & coefficients
print("Intercept:", model.intercept_)
print("Coefficients:")
for feature, coef in zip(X.columns, model.coef_):
    print(f"{feature}: {coef}")

# Display first 5 actual vs predicted
comparison = pd.DataFrame({
    'Actual': y[:5],
    'Predicted': y_pred[:5]
})

print("\nFirst 5 Actual vs Predicted:")
print(comparison)

#Task 2: Model Evaluation



# Calculate metrics
mae = mean_absolute_error(y, y_pred)
rmse = np.sqrt(mean_squared_error(y, y_pred))
r2 = r2_score(y, y_pred)

print("\nModel Evaluation Metrics:")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)

# Explanation:
# MAE shows the average absolute error in predictions (lower is better).
# RMSE penalizes large errors more than MAE, giving insight into variance of errors.
# R² indicates how well the model explains variance; closer to 1 means better fit.

#Task 3: Residual Analysis


# Compute residuals
residuals = y - y_pred

# Plot histogram
plt.figure()
plt.hist(residuals, bins=10)
plt.title("Residuals Distribution")
plt.xlabel("Residuals")
plt.ylabel("Frequency")
plt.show()

# Explanation:
# A residual is the difference between actual and predicted value.
# If the histogram is roughly symmetric and centered around 0, the model is well-fitted.
# Skewness or wide spread indicates bias or high prediction errors.