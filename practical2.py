import numpy as np
import matplotlib.pyplot as plt

# Given datasets

X1 = [1, 2, 3, 4, 5, 4, 3, 2, 4, 5,
      1, 2, 3, 4, 2, 5, 1, 3, 2, 1,
      2, 1, 1, 1, 2]

X2 = [1, 2, 3, 4, 5, 4, 3, 2, 4, 5,
      1, 2, 3, 4, 2, 5, 1, 3, 2, 1,
      2, 1, 1, 1, 2000]

X3 = [1, 2, 3, 10, 20, 30, 100, 200,
      300, 1000, 2000, 3000]


# Function
def calculate_statistics(data):

    mean = np.mean(data)
    variance = np.var(data)
    standard_deviation = np.std(data)

    return mean, variance, standard_deviation


# Calculate statistics
for name, data in [("X1", X1), ("X2", X2), ("X3", X3)]:

    mean, variance, standard_deviation = calculate_statistics(data)

    print("\nDataset:", name)
    print("Mean =", mean)
    print("Variance =", variance)
    print("Standard Deviation =", standard_deviation)


# Histogram
plt.hist(X1, bins=range(1, 8), edgecolor="black")

plt.xlabel("Data Values")
plt.ylabel("Frequency")
plt.title("Histogram of X1")

plt.show()
