# Experiment No. 10
# Calculate Within-Class, Between-Class
# and Total Scatter using Iris Dataset

import numpy as np

from sklearn.datasets import load_iris


# ---------------------------------------------------
# Load Iris Dataset
# ---------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target


# ---------------------------------------------------
# Overall Mean
# ---------------------------------------------------

overall_mean = np.mean(
    X,
    axis=0
)


# ---------------------------------------------------
# Number of Features
# ---------------------------------------------------

n_features = X.shape[1]


# ---------------------------------------------------
# Initialize Scatter Matrices
# ---------------------------------------------------

SW = np.zeros(
    (n_features, n_features)
)

SB = np.zeros(
    (n_features, n_features)
)


# ---------------------------------------------------
# Calculate Within-Class and
# Between-Class Scatter
# ---------------------------------------------------

for class_label in np.unique(y):

    # Select class data
    X_class = X[
        y == class_label
    ]

    # Mean of class
    class_mean = np.mean(
        X_class,
        axis=0
    )

    # Number of samples
    n_class = X_class.shape[0]


    # ------------------------------------------------
    # Within-Class Scatter
    # ------------------------------------------------

    X_centered = (
        X_class - class_mean
    )

    SW += (
        X_centered.T
        @
        X_centered
    )


    # ------------------------------------------------
    # Between-Class Scatter
    # ------------------------------------------------

    mean_difference = (
        class_mean - overall_mean
    ).reshape(-1, 1)

    SB += (
        n_class
        *
        (
            mean_difference
            @
            mean_difference.T
        )
    )


# ---------------------------------------------------
# Total Scatter
# ---------------------------------------------------

X_centered_total = (
    X - overall_mean
)

ST = (
    X_centered_total.T
    @
    X_centered_total
)


# ---------------------------------------------------
# Display Results
# ---------------------------------------------------

np.set_printoptions(
    precision=4,
    suppress=True
)

print("Overall Mean:")
print(overall_mean)


print("\nWithin-Class Scatter Matrix (SW):")
print(SW)


print("\nBetween-Class Scatter Matrix (SB):")
print(SB)


print("\nTotal Scatter Matrix (ST):")
print(ST)


# ---------------------------------------------------
# Verify ST = SW + SB
# ---------------------------------------------------

print(
    "\nVerification of ST = SW + SB:"
)

if np.allclose(
    ST,
    SW + SB
):

    print(
        "ST = SW + SB is TRUE"
    )

else:

    print(
        "ST = SW + SB is FALSE"
    )


# ---------------------------------------------------
# Trace / Scalar Scatter Values
# ---------------------------------------------------

print(
    "\nScalar Scatter Values (Trace):"
)

print(
    "Within-Class Scatter :",
    np.trace(SW)
)

print(
    "Between-Class Scatter:",
    np.trace(SB)
)

print(
    "Total Scatter :",
    np.trace(ST)
)
