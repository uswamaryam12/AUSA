import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
data = pd.read_csv("dataset.csv")

# Preprocess the dataset
# Drop unnecessary columns like Name
data = data.drop(columns=["Name"])

# Convert categorical variables to numerical using one-hot encoding
data_encoded = pd.get_dummies(data, columns=["Initial Continent", "Initial Climate", "Final Continent", "Final Climate"])

# Split the dataset into features and target variables
X = data_encoded.drop(columns=["Impact"])
y = data_encoded["Impact"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Build the linear regression model
model = LinearRegression()

# Train the model
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)

# Calculate mean squared error and R^2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print("Mean Squared Error:", mse)
print("R^2 Score:", r2)

# Combine actual and predicted values for comparison
results_df = pd.DataFrame({'Actual': y_test, 'Predicted': y_pred})

# Print the results
print("Actual vs Predicted Impact:")
print(results_df)

# Plot actual vs predicted values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel('Actual Impact')
plt.ylabel('Predicted Impact')
plt.title('Actual vs Predicted Impact')
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red')  # Line of best fit
plt.show()

# Plot residuals
residuals = y_test - y_pred
plt.scatter(y_pred, residuals, alpha=0.5)
plt.xlabel('Predicted Impact')
plt.ylabel('Residuals')
plt.title('Residuals Plot')
plt.axhline(y=0, color='red')
plt.show()
