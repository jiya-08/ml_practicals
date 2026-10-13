# Experiment No. 8
# Decision Tree Classification
# Iris Dataset

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.tree import (
    DecisionTreeClassifier,
    plot_tree
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

import matplotlib.pyplot as plt


# ---------------------------------------------------
# Load Iris Dataset
# ---------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target


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
# Create Decision Tree Model
# ---------------------------------------------------

model = DecisionTreeClassifier(
    criterion="entropy",
    random_state=42
)


# ---------------------------------------------------
# Train Model
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

print("Decision Tree Accuracy:")
print(
    "{:.2f}%".format(
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
# Display Decision Tree
# ---------------------------------------------------

plt.figure(
    figsize=(15, 10)
)

plot_tree(
    model,
    feature_names=iris.feature_names,
    class_names=iris.target_names,
    filled=True
)

plt.title(
    "Decision Tree Classifier - Iris Dataset"
)

plt.show()
