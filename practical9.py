# Experiment No. 9
# Neural Network Classification
# Iris Dataset

import numpy as np

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.neural_network import MLPClassifier

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ---------------------------------------------------
# Load Iris Dataset
# ---------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target


# ---------------------------------------------------
# Split Dataset
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------
# Standardize Features
# ---------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(
    X_train
)

X_test = scaler.transform(
    X_test
)


# ---------------------------------------------------
# Create Neural Network
# ---------------------------------------------------

model = MLPClassifier(
    hidden_layer_sizes=(10, 10),
    activation="relu",
    solver="adam",
    max_iter=1000,
    random_state=42
)


# ---------------------------------------------------
# Train Neural Network
# ---------------------------------------------------

model.fit(
    X_train,
    y_train
)


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

y_pred = model.predict(
    X_test
)


# ---------------------------------------------------
# Accuracy
# ---------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print(
    "Neural Network Accuracy: {:.2f}%".format(
        accuracy * 100
    )
)


# ---------------------------------------------------
# Classification Report
# ---------------------------------------------------

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=iris.target_names
    )
)


# ---------------------------------------------------
# Confusion Matrix
# ---------------------------------------------------

print("\nConfusion Matrix:")

cm = confusion_matrix(
    y_test,
    y_pred
)

print(cm)


# ---------------------------------------------------
# Predict New Flower
# ---------------------------------------------------

new_flower = np.array([
    [5.1, 3.5, 1.4, 0.2]
])

new_flower_scaled = scaler.transform(
    new_flower
)

prediction = model.predict(
    new_flower_scaled
)

print(
    "\nPredicted Class:",
    iris.target_names[prediction[0]]
)
