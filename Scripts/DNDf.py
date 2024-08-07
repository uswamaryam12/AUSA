import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd

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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Reshape the input data for GRU
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], X_train_scaled.shape[1], 1))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], X_test_scaled.shape[1], 1))
# Define the DNN model
def create_dnn_model(input_shape):
    model = tf.keras.Sequential([
        Dense(128, activation='relu', input_shape=input_shape),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dropout(0.2),
        Dense(32, activation='relu'),
        Dropout(0.2),
        Dense(1, activation='sigmoid')
    ])
    return model

# Train the DNN model
dnn_model = create_dnn_model(X_train.shape[1:])
dnn_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
dnn_model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Make predictions using DNN model
dnn_predictions = dnn_model.predict(X_test)
dnn_predictions_binary = (dnn_predictions > 0.5).astype(int)

# Define the decision forest model
forest_model = RandomForestClassifier(n_estimators=100)

# Train the decision forest model
forest_model.fit(X_train, y_train)

# Make predictions using decision forest model
forest_predictions = forest_model.predict(X_test)

# Combine predictions from DNN and decision forest
combined_predictions = (dnn_predictions_binary + forest_predictions) / 2
combined_predictions_binary = (combined_predictions > 0.5).astype(int)

# Calculate accuracy
dnn_accuracy = accuracy_score(y_test, dnn_predictions_binary)
forest_accuracy = accuracy_score(y_test, forest_predictions)
combined_accuracy = accuracy_score(y_test, combined_predictions_binary)

print("DNN Model Accuracy:", dnn_accuracy)
print("Decision Forest Model Accuracy:", forest_accuracy)
print("Combined Model Accuracy:", combined_accuracy)
