import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

# Load the dataset
data = pd.read_csv("dataset.csv")

# Preprocess the dataset
data = data.drop(columns=["Name"])
data_encoded = pd.get_dummies(data, columns=["Initial Continent", "Initial Climate", "Final Continent", "Final Climate"])

# Split the dataset into features and target variables
X = data_encoded.drop(columns=["Impact"])
y = data_encoded["Impact"]

# Check class distribution
class_distribution = y.value_counts()
print("Class Distribution:")
print(class_distribution)

# Identify classes with very few instances
minority_classes = class_distribution[class_distribution < 5].index.tolist()
print("Minority Classes:")
print(minority_classes)

# Handle class imbalance using SMOTE with adjusted k_neighbors
if minority_classes:
    smote_k_neighbors = min(3, min(class_distribution[minority_classes]))
else:
    smote_k_neighbors = 5

smote = SMOTE(random_state=42, k_neighbors=smote_k_neighbors)  # Adjust k_neighbors as needed
X_resampled, y_resampled = smote.fit_resample(X, y)

# Split the resampled dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Create a pipeline with polynomial features and logistic regression
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures(degree=2)),  # Degree can be tuned
    ('logreg', LogisticRegression(max_iter=300, random_state=42, multi_class='multinomial', solver='lbfgs'))
])

# Hyperparameter tuning using GridSearchCV
param_grid = {
    'poly__degree': [1, 2, 3],  # Try different polynomial degrees
    'logreg__C': [0.01, 0.1, 1, 10, 100]  # Regularization strength
}

grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Best model from grid search
best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)

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
y_pred_prob = best_model.predict_proba(X_test)
for i in range(len(best_model.named_steps['logreg'].classes_)):
    y_test_bin = (y_test == best_model.named_steps['logreg'].classes_[i]).astype(int)
    roc_auc_dict[best_model.named_steps['logreg'].classes_[i]] = roc_auc_score(y_test_bin, y_pred_prob[:, i])
    fpr, tpr, _ = roc_curve(y_test_bin, y_pred_prob[:, i])
    plt.plot(fpr, tpr, label=f"Class {best_model.named_steps['logreg'].classes_[i]} (area = {roc_auc_dict[best_model.named_steps['logreg'].classes_[i]]:.2f})")

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
