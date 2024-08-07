import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import griddata

# Load the dataset
data = pd.read_csv("output.csv")

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

# Reshape the input data for LSTM
X_train_reshaped = np.reshape(X_train_scaled, (X_train_scaled.shape[0], 1, X_train_scaled.shape[1]))
X_test_reshaped = np.reshape(X_test_scaled, (X_test_scaled.shape[0], 1, X_test_scaled.shape[1]))

# Build the LSTM model
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(1, X_train_scaled.shape[1])))
model.add(Dropout(0.2))
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(units=50))
model.add(Dropout(0.2))
model.add(Dense(units=1))  # Output layer with 1 neuron for Impact

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train_reshaped, y_train, epochs=100, batch_size=32, validation_data=(X_test_reshaped, y_test), verbose=1)

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

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(predictions, columns=["Predicted_Impact"])

# Reset index for y_test for concatenation
y_test.reset_index(drop=True, inplace=True)

# Concatenate actual and predicted values
results_df = pd.concat([y_test, predictions_df], axis=1)

# Print the results
print("Actual vs Predicted Impact:")
print(results_df)

# Perform PCA on the feature set
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Convert predictions and actual values to numpy arrays for plotting
predictions_np = predictions.flatten()
y_test_np = y_test.values.flatten()

# Create a DataFrame with PCA components and predictions
pca_df = pd.DataFrame(data=X_test_pca, columns=["PC1", "PC2"])
pca_df["Actual_Impact"] = y_test_np
pca_df["Predicted_Impact"] = predictions_np

# Create a mesh grid for the wireframe plot
grid_x, grid_y = np.meshgrid(
    np.linspace(pca_df["PC1"].min(), pca_df["PC1"].max(), 100),
    np.linspace(pca_df["PC2"].min(), pca_df["PC2"].max(), 100)
)

# Interpolate the predicted impacts for the mesh grid
grid_z = griddata((pca_df["PC1"], pca_df["PC2"]), pca_df["Predicted_Impact"], (grid_x, grid_y), method='cubic')

# Plotting the 3D wireframe plot
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Wireframe plot for predicted impacts
ax.plot_wireframe(grid_x, grid_y, grid_z, color='r')

# Labels and title
ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Predicted Impact')
ax.set_title('3D Wireframe Plot of Predicted Impact')

# Show the plot
plt.show()
