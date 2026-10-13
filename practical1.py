import statistics
import matplotlib.pyplot as plt

# Datasets
X1 = [1, 2, 3, 4, 5, 4, 3, 2, 4, 5,
      1, 2, 3, 4, 2, 5, 1, 3, 2, 1,
      2, 1, 1, 1, 2]

X2 = [1, 2, 3, 4, 5, 4, 3, 2, 4, 5,
      1, 2, 3, 4, 2, 5, 1, 3, 2, 1,
      2, 1, 1, 1, 2000]

X3 = [1, 2, 3, 10, 20, 30, 100, 200,
      300, 1000, 2000, 3000]


def calculate_statistics(data):
    mean = statistics.mean(data)
    median = statistics.median(data)
    mode = statistics.multimode(data)

    return mean, median, mode


# Calculate statistics
for name, data in [("X1", X1), ("X2", X2), ("X3", X3)]:

    mean, median, mode = calculate_statistics(data)

    print("\nDataset:", name)
    print("Mean:", mean)
    print("Median:", median)
    print("Mode:", mode)


# Histogram
X = [1, 3, 2, 4, 5, 6, 4, 3, 2, 4,
     5, 3, 1, 2, 3, 2, 3, 1, 4]

mean = statistics.mean(X)
median = statistics.median(X)
mode = statistics.mode(X)

plt.hist(X, bins=range(1, 8),
         edgecolor='black',
         align='left')

plt.axvline(mean, linestyle='--', label='Mean')
plt.axvline(median, linestyle='-.', label='Median')
plt.axvline(mode, linestyle=':', label='Mode')

plt.xlabel("Data Values")
plt.ylabel("Frequency")
plt.title("Histogram with Mean, Median and Mode")
plt.legend()
plt.show()
