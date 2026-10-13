import numpy as np


# Given vectors
A = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

B = np.array([1, 3, 5, 7, 9, 7, 5, 3, 1, 0])

X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])

Y = np.array([1, 3, 5, 7, 9, 7, 5, 3, 1, 0])


# Cosine Similarity
def cosine_similarity(A, B):

    return np.dot(A, B) / (
        np.linalg.norm(A) * np.linalg.norm(B)
    )


# Euclidean Distance
def euclidean_distance(A, B):

    return np.sqrt(np.sum((A - B) ** 2))


# Calculate
cosine_AB = cosine_similarity(A, B)
cosine_XY = cosine_similarity(X, Y)

euclidean_AB = euclidean_distance(A, B)
euclidean_XY = euclidean_distance(X, Y)


# Display results
print("Cosine Similarity (A, B):", cosine_AB)
print("Cosine Similarity (X, Y):", cosine_XY)

print("Euclidean Distance (A, B):", euclidean_AB)
print("Euclidean Distance (X, Y):", euclidean_XY)
