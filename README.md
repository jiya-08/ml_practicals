# Machine Learning Practicals

Repository containing 12 structured practical experiments for the Machine Learning course. Each experiment is hosted on its dedicated feature branch with program code, execution output image, and a brief overview.

---

## Branching Strategy

- **`main`**: Production documentation branch with repository overview and experiments directory table.
- **`dev`**: Active integration branch where feature branches submit Pull Requests.
- **`feature/exp<n>`**: Dedicated branches for each experiment (`feature/exp1` through `feature/exp12`).

---

## Experiments Index

| Exp No. | Experiment Title | Feature Branch | Key Concepts | Overview |
| :---: | :--- | :---: | :--- | :--- |
| **1** | **Measures of Central Tendency and Dispersion** | [`feature/exp1`](https://github.com/jiya-08/ml_practicals/tree/feature/exp1) | `Mean, Median, Mode, Variance, Standard Deviation, Histogram` | Calculates central tendency (mean, median, mode) and dispersion (variance, standard deviation) for discrete datasets, and visualizes the frequency distribution using a histogram. |
| **2** | **Normal (Gaussian) Distribution Analysis** | [`feature/exp2`](https://github.com/jiya-08/ml_practicals/tree/feature/exp2) | `Normal Distribution, Gaussian PDF, Variance, Data Distribution` | Computes probability density values and statistical parameters (mean, standard deviation) for normal distribution, plotting normal distribution curves and histogram distributions. |
| **3** | **Principal Component Analysis (PCA)** | [`feature/exp3`](https://github.com/jiya-08/ml_practicals/tree/feature/exp3) | `Dimensionality Reduction, Feature Scaling, Eigenvectors, PCA Projection` | Performs dimensionality reduction on the 4-dimensional Iris dataset using PCA, transforming features into 2 principal components with scatter plot visualization. |
| **4** | **Vector Operations and Distance Metrics** | [`feature/exp4`](https://github.com/jiya-08/ml_practicals/tree/feature/exp4) | `Euclidean Distance, Manhattan Distance, Cosine Similarity, Vector Norms` | Calculates fundamental vector operations and metric distances between feature vectors, including Euclidean distance, Manhattan distance, and Cosine similarity. |
| **5** | **Simple Linear Regression** | [`feature/exp5`](https://github.com/jiya-08/ml_practicals/tree/feature/exp5) | `Linear Regression, Least Squares, MSE, R2 Score, Best Fit Line` | Fits an Ordinary Least Squares (OLS) Linear Regression model to predict sales from advertising spend, calculating slope, intercept, Mean Squared Error (MSE), and R² score. |
| **6** | **Logistic Regression Classification** | [`feature/exp6`](https://github.com/jiya-08/ml_practicals/tree/feature/exp6) | `Binary/Multiclass Classification, Confusion Matrix, Precision, Recall, F1` | Trains a multiclass Logistic Regression classifier on the Iris dataset, evaluating accuracy, precision, recall, F1-score, and displaying a confusion matrix heatmap. |
| **7** | **K-Nearest Neighbors (KNN) Classification** | [`feature/exp7`](https://github.com/jiya-08/ml_practicals/tree/feature/exp7) | `Instance-based Learning, K-Nearest Neighbors, Feature Scaling, Hyperparameter K` | Implements KNN classification on the Iris dataset, evaluating model accuracy across varying K values (K=1, 3, 5, 7, 9) and generating detailed classification reports. |
| **8** | **Decision Tree Classification** | [`feature/exp8`](https://github.com/jiya-08/ml_practicals/tree/feature/exp8) | `Decision Trees, Gini Impurity, Information Gain, Tree Pruning` | Builds and evaluates Decision Tree classifiers on the Iris dataset using Gini and Entropy impurity criteria, outputting confusion matrix and classification metrics. |
| **9** | **Neural Network (Multi-Layer Perceptron) Classification** | [`feature/exp9`](https://github.com/jiya-08/ml_practicals/tree/feature/exp9) | `Artificial Neural Networks, MLPClassifier, Backpropagation, Hidden Layers` | Builds a Multi-Layer Perceptron (MLP) neural network classifier to learn nonlinear decision boundaries on the Iris dataset and evaluates classification accuracy. |
| **10** | **Within-Class, Between-Class and Total Scatter Matrices** | [`feature/exp10`](https://github.com/jiya-08/ml_practicals/tree/feature/exp10) | `Scatter Matrices, Linear Discriminant Analysis (LDA), Multiclass Separability` | Computes Within-Class Scatter (SW), Between-Class Scatter (SB), and Total Scatter (ST) matrices on the Iris dataset, mathematically verifying the identity ST = SW + SB. |
| **11** | **K-Means Clustering** | [`feature/exp11`](https://github.com/jiya-08/ml_practicals/tree/feature/exp11) | `Unsupervised Learning, K-Means Clustering, Centroid Update, Distance Measures` | Implements unsupervised K-Means clustering algorithm on the Iris dataset, comparing clustering performance across cluster counts (K=2 to 6) using Euclidean and Manhattan distances. |
| **12** | **Introduction to Machine Learning Libraries (NumPy, Pandas, Scikit-learn)** | [`feature/exp12`](https://github.com/jiya-08/ml_practicals/tree/feature/exp12) | `NumPy Arrays, Pandas DataFrames, Scikit-learn LinearRegression, MSE` | Demonstrates essential Python data science libraries: array operations and statistics with NumPy, DataFrame tabular structures with Pandas, and linear model fitting with Scikit-learn. |

---

## How to Explore

Clone the repository and switch to any experiment branch:

```bash
# Clone the repository
git clone https://github.com/jiya-08/ml_practicals.git

# Switch to the development branch
git checkout dev

# Check out a specific experiment branch
git checkout feature/exp1
```

Each experiment branch contains:
- `practical<n>.py`: Executable Python program
- `practical<n>.ipynb`: Interactive Jupyter Notebook
- `output.png`: Execution output / plot screenshot
- `README.md`: Brief overview and output preview
