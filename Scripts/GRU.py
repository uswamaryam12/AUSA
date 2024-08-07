import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from keras.models import Sequential
from keras.layers import GRU, Dense, Dropout

# Load the dataset
data = pd.read_csv("output.csv")

# Print the first few rows of the dataset
print(data.head())

# Check the columns of the dataset
print(data.columns)

# Ensure all expected columns are present
expected_columns = ["Name", "Skin Tone", "Initial Continent", "Initial Climate", "Initial Radiations",
                    "Initial Oxygen", "Initial Nitrogen", "Final Continent", "Final Climate",
                    "Final Radiations", "Final Oxygen", "Final Nitrogen", "Duration", "Impact"]

missing_columns = [col for col in expected_columns if col not in data.columns]
if missing_columns:
    print(f"Missing columns in the dataset: {missing_columns}")
else:
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

    # Build the GRU model
    model = Sequential()
    model.add(GRU(units=50, activation='relu', input_shape=(X_train_reshaped.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))  # Output layer with 1 neuron for Impact

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train_reshaped, y_train, epochs=300, batch_size=32, validation_data=(X_test_reshaped, y_test), verbose=1)

    # Plot training and validation loss
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Evaluate the model
    loss = model.evaluate(X_test_reshaped, y_test)
    print("Test Loss:", loss)

    # Make predictions
    predictions = model.predict(X_test_reshaped)

    # Plot actual vs predicted values
    plt.scatter(y_test, predictions)
    plt.xlabel('Actual Impact')
    plt.ylabel('Predicted Impact')
    plt.title('Actual vs Predicted Impact')
    plt.show()
