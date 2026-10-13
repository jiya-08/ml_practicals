# Experiment No. 7
# K-Nearest Neighbors (KNN) Classification
# Iris Dataset

import numpy as np

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report


# ---------------------------------------------------
# Load Iris Dataset
# ---------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target

print("Dataset Shape:", X.shape)


# ---------------------------------------------------
# Split Dataset
# 70% Training and 30% Testing
# ---------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.30,
    random_state=42,
    stratify=y
)


# ---------------------------------------------------
# Feature Scaling
# ---------------------------------------------------

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)


# ---------------------------------------------------
# Try Different Values of K
# ---------------------------------------------------

k_values = [1, 3, 5, 7, 9]

print("\nKNN Accuracy Results")
print("--------------------")

for k in k_values:

    model = KNeighborsClassifier(n_neighbors=k)

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)

    print(
        "K =", k,
        "Accuracy = {:.2f}%".format(accuracy * 100)
    )


# ---------------------------------------------------
# Final KNN Model
# ---------------------------------------------------

k = 5

model = KNeighborsClassifier(
    n_neighbors=k
)

model.fit(X_train, y_train)


# ---------------------------------------------------
# Prediction
# ---------------------------------------------------

y_pred = model.predict(X_test)


# ---------------------------------------------------
# Accuracy
# ---------------------------------------------------

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nFinal K =", k)
print("Accuracy = {:.2f}%".format(accuracy * 100))


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
# Predict a New Flower
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
