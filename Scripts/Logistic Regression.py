import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

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

# Build the logistic regression model
model = LogisticRegression(max_iter=300, random_state=42, multi_class='multinomial', solver='lbfgs')

# Train the model
model.fit(X_train_scaled, y_train)

# Evaluate the model
y_pred = model.predict(X_test_scaled)

# Calculate accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate a classification report
report = classification_report(y_test, y_pred)
print("Classification Report:")
print(report)

# Generate a confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Calculate ROC AUC score for each class
roc_auc_dict = {}
for i in range(len(model.classes_)):
    y_test_bin = (y_test == model.classes_[i]).astype(int)
    y_pred_prob = model.predict_proba(X_test_scaled)[:, i]
    roc_auc_dict[model.classes_[i]] = roc_auc_score(y_test_bin, y_pred_prob)
    fpr, tpr, _ = roc_curve(y_test_bin, y_pred_prob)
    plt.plot(fpr, tpr, label=f'Class {model.classes_[i]} (area = {roc_auc_dict[model.classes_[i]]:.2f})')

# Plot ROC curve
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

# Print ROC AUC scores
for cls, score in roc_auc_dict.items():
    print(f'Class {cls} ROC AUC Score: {score}')

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
plt.show()
