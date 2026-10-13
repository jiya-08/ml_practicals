# Experiment No. 11
# K-Means Clustering using Iris Dataset

import numpy as np

from collections import Counter

from sklearn.datasets import load_iris

from sklearn.model_selection import train_test_split

from sklearn.cluster import KMeans

from sklearn.metrics import accuracy_score


# ---------------------------------------------------
# Load Iris Dataset
# ---------------------------------------------------

iris = load_iris()

X = iris.data
y = iris.target


# ---------------------------------------------------
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
# Function to Convert Cluster Numbers
# into Actual Class Labels
# ---------------------------------------------------

def calculate_accuracy(
    train_clusters,
    y_train,
    test_clusters,
    y_test
):

    cluster_to_class = {}


    for cluster in np.unique(
        train_clusters
    ):

        class_values = y_train[
            train_clusters == cluster
        ]

        if len(class_values) > 0:

            majority_class = Counter(
                class_values
            ).most_common(1)[0][0]

            cluster_to_class[
                cluster
            ] = majority_class


    predicted_labels = []

    for cluster in test_clusters:

        predicted_labels.append(
            cluster_to_class[cluster]
        )


    accuracy = accuracy_score(
        y_test,
        predicted_labels
    )

    return accuracy


# ---------------------------------------------------
# Standard K-Means
# Euclidean Distance
# ---------------------------------------------------

print(
    "K-MEANS CLUSTERING USING EUCLIDEAN DISTANCE"
)

print("-" * 55)


k_values = [
    2, 3, 4, 5, 6
]


for k in k_values:

    model = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=10
    )


    train_clusters = model.fit_predict(
        X_train
    )


    test_clusters = model.predict(
        X_test
    )


    accuracy = calculate_accuracy(
        train_clusters,
        y_train,
        test_clusters,
        y_test
    )


    print(
        "K =",
        k,
        "Accuracy = {:.2f}%".format(
            accuracy * 100
        )
    )


# ---------------------------------------------------
# Custom K-Means
# ---------------------------------------------------

def custom_kmeans(
    X,
    k,
    distance="euclidean",
    max_iter=100,
    random_state=42
):

    np.random.seed(
        random_state
    )


    # Select initial centroids
    random_indices = np.random.choice(
        len(X),
        k,
        replace=False
    )


    centroids = X[
        random_indices
    ].copy()


    # Iterations
    for iteration in range(
        max_iter
    ):


        # -------------------------------------------
        # Calculate Distance
        # -------------------------------------------

        if distance == "euclidean":

            distances = np.sqrt(
                (
                    (
                        X[:, np.newaxis]
                        -
                        centroids[np.newaxis, :]
                    ) ** 2
                ).sum(axis=2)
            )


        elif distance == "manhattan":

            distances = np.abs(
                X[:, np.newaxis]
                -
                centroids[np.newaxis, :]
            ).sum(axis=2)


        # -------------------------------------------
        # Assign each point to nearest centroid
        # -------------------------------------------

        clusters = np.argmin(
            distances,
            axis=1
        )


        # -------------------------------------------
        # Calculate New Centroids
        # -------------------------------------------

        new_centroids = []


        for i in range(k):

            cluster_points = X[
                clusters == i
            ]


            if len(cluster_points) > 0:

                new_centroid = (
                    cluster_points.mean(
                        axis=0
                    )
                )

            else:

                new_centroid = centroids[i]


            new_centroids.append(
                new_centroid
            )


        new_centroids = np.array(
            new_centroids
        )


        # -------------------------------------------
        # Check Convergence
        # -------------------------------------------

        if np.allclose(
            centroids,
            new_centroids
        ):

            break


        centroids = new_centroids


    return centroids, clusters


# ---------------------------------------------------
# Predict Clusters
# ---------------------------------------------------

def predict_clusters(
    X,
    centroids,
    distance="euclidean"
):

    if distance == "euclidean":

        distances = np.sqrt(
            (
                (
                    X[:, np.newaxis]
                    -
                    centroids[np.newaxis, :]
                ) ** 2
            ).sum(axis=2)
        )

    else:

        distances = np.abs(
            X[:, np.newaxis]
            -
            centroids[np.newaxis, :]
        ).sum(axis=2)


    return np.argmin(
        distances,
        axis=1
    )


# ---------------------------------------------------
# Compare Euclidean and Manhattan Distance
# ---------------------------------------------------

print(
    "\nDISTANCE MEASURE COMPARISON"
)

print("-" * 55)


for k in [2, 3, 4, 5]:

    for distance in [
        "euclidean",
        "manhattan"
    ]:

        centroids, train_clusters = custom_kmeans(
            X_train,
            k,
            distance=distance
        )


        test_clusters = predict_clusters(
            X_test,
            centroids,
            distance=distance
        )


        accuracy = calculate_accuracy(
            train_clusters,
            y_train,
            test_clusters,
            y_test
        )


        print(
            "K =",
            k,
            "| Distance =",
            distance,
            "| Accuracy = {:.2f}%".format(
                accuracy * 100
            )
        )
