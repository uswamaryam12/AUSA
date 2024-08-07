import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import plotly.graph_objects as go

# Load the dataset
data = pd.read_csv("output.csv")

# Preprocess the dataset
# Drop unnecessary columns like Name
data = data.drop(columns=["Name"])

# Convert categorical variables to numerical using one-hot encoding
data_encoded = pd.get_dummies(data,
                              columns=["Initial Continent", "Initial Climate", "Final Continent", "Final Climate"])

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

# Initialize lists to store training and validation losses
train_losses = []
val_losses = []

# Train the model
for epoch in range(300):
    history = model.fit(X_train_reshaped, y_train, epochs=1, batch_size=32, validation_data=(X_test_reshaped, y_test),
                        verbose=0)

    # Append the training and validation losses
    train_losses.append(history.history['loss'][0])
    val_losses.append(history.history['val_loss'][0])

# Make predictions
predictions = model.predict(X_test_reshaped)

# Convert predictions to DataFrame
predictions_df = pd.DataFrame(predictions, columns=["Predicted_Impact"])

# Perform PCA on the feature set
pca = PCA(n_components=3)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)

# Convert predictions and actual values to numpy arrays for plotting
predictions_np = predictions.flatten()
y_test_np = y_test.values.flatten()

# Create a DataFrame with PCA components and predictions
pca_df = pd.DataFrame(data=X_test_pca, columns=["PC1", "PC2", "PC3"])
pca_df["Actual_Impact"] = y_test_np
pca_df["Predicted_Impact"] = predictions_np

# Create a 3D scatter plot using Plotly
fig = go.Figure(data=[go.Scatter3d(
    x=pca_df["PC1"],
    y=pca_df["PC2"],
    z=pca_df["PC3"],
    mode='markers',
    marker=dict(
        size=5,
        color=pca_df["Predicted_Impact"],
        colorscale='Viridis',  # choose a colorscale
        opacity=0.8
    )
)])

# Update plot layout
fig.update_layout(
    title='3D Scatter Plot of Predicted Impact',
    scene=dict(
        xaxis_title='Principal Component 1',
        yaxis_title='Principal Component 2',
        zaxis_title='Principal Component 3'
    )
)

# Show the plot
fig.show()

# Plot the gradient descent curve
plt.plot(train_losses, label='Training Loss')
plt.plot(val_losses, label='Validation Loss')
plt.title('Gradient Descent')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()
