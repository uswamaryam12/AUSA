import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from keras.models import Sequential
from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

# Load the dataset
data = pd.read_csv("dataset.csv")

# Preprocess the dataset
# Drop unnecessary columns like Name
data = data.drop(columns=["Name"])

# Convert categorical variables to numerical using one-hot encoding
data_encoded = pd.get_dummies(data, columns=["Initial Continent", "Initial Climate", "Final Continent", "Final Climate"])

# Split the dataset into features and target variables
X = data_encoded.drop(columns=["Impact"])
y = data_encoded[["Impact"]]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the input data for CNN
X_train_reshaped = X_train_scaled.reshape((X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = X_test_scaled.reshape((X_test_scaled.shape[0], X_test_scaled.shape[1], 1))

# Build the CNN model
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(X_train_scaled.shape[1], 1)))
model.add(MaxPooling1D(pool_size=2))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))  # Output layer with 1 neuron for Impact

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, verbose=1)

# Evaluate the model
loss = model.evaluate(X_test_reshaped, y_test)
print("Test Loss:", loss)

# Make predictions
predictions = model.predict(X_test_reshaped)

# Calculate R-squared score
r2 = r2_score(y_test, predictions)
print("R-squared Score:", r2)


# Convert predictions to DataFrame
predictions_df = pd.DataFrame(predictions, columns=["Predicted_Impact"])

# Reset index for y_test for concatenation
y_test.reset_index(drop=True, inplace=True)

# Concatenate actual and predicted values
results_df = pd.concat([y_test, predictions_df], axis=1)

# Print the results
print("Actual vs Predicted Impact:")
print(results_df)