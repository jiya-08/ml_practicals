# Experiment No. 12
# Introduction to Machine Learning Libraries (NumPy, Pandas, Scikit-learn)

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. NumPy Library Demonstration
print("NumPy Library:")
arr = np.array([10, 20, 30, 40, 50])
print("Array:", arr)
print("Mean:", np.mean(arr))

# 2. Pandas Library Demonstration
print("\nPandas Library:")
data = {
    'Hours': [1, 2, 3, 4, 5],
    'Marks': [20, 40, 50, 70, 80]
}
df = pd.DataFrame(data)
print(df)
print("Average Marks:", df['Marks'].mean())

# 3. Scikit-learn Library Demonstration
print("\nScikit-learn Library:")
X = df[['Hours']]
y = df['Marks']

model = LinearRegression()
model.fit(X, y)

pred = model.predict([[6]])
print("Predicted marks for 6 hours of study:", pred[0])

y_pred = model.predict(X)
print("Mean Squared Error:", mean_squared_error(y, y_pred))
