import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


# Given data
X = np.array([
    70, 80, 90, 100, 110,
    120, 130, 140, 150, 160
])

Y = np.array([
    7, 7, 8, 9, 12,
    12, 15, 14, 13, 17
])


# Reshape X
X_reshaped = X.reshape(-1, 1)


# Create model
model = LinearRegression()


# Train model
model.fit(X_reshaped, Y)


# Model parameters
w0 = model.intercept_
w1 = model.coef_[0]

print("Intercept (w0):", w0)
print("Slope (w1):", w1)


# Predict Y for X = 210
x_new = np.array([[210]])

y_pred = model.predict(x_new)

print("Predicted Y for X = 210:", y_pred[0])


# Predict values for original data
Y_pred = model.predict(X_reshaped)


# Evaluate model
mse = mean_squared_error(Y, Y_pred)
r2 = r2_score(Y, Y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)


# Plot actual data
plt.scatter(
    X,
    Y,
    label="Actual Data"
)


# Regression line
plt.plot(
    X,
    Y_pred,
    label="Regression Line"
)


# Prediction point
plt.scatter(
    210,
    y_pred[0],
    marker='*',
    s=150,
    label="Prediction at X=210"
)


plt.xlabel("Advertisement Spending")
plt.ylabel("Increase in Unit Sales")
plt.title("Simple Linear Regression")

plt.legend()
plt.grid(True)

plt.show()
