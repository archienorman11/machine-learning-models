"""
This script performs linear regression using gradient descent to predict an outcome based on two features.
The script loads a dataset containing the features and the outcome, normalizes the feature values, and then
applies gradient descent to learn the optimal weights (theta) for the model. The cost function is calculated 
during each iteration of the gradient descent to monitor the model's performance. Finally, a plot of the cost 
function is displayed over the number of iterations to visualize the optimization process.
"""


import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, List

def feature_normalize(X: np.ndarray) -> Tuple[np.ndarray, List[float], List[float]]:
    mean_r = []
    std_r = []

    X_norm = X.copy()
    n_c = X.shape[1]

    for i in range(n_c):
        m = np.mean(X[:, i])
        s = np.std(X[:, i])
        mean_r.append(m)
        std_r.append(s)
        X_norm[:, i] = (X_norm[:, i] - m) / s

    return X_norm, mean_r, std_r

def compute_cost(X: np.ndarray, y: np.ndarray, theta: np.ndarray) -> float:
    m = y.size
    predictions = X.dot(theta)
    sq_errors = (predictions - y)
    J = (1.0 / (2 * m)) * sq_errors.T.dot(sq_errors)
    return J

def gradient_descent(X: np.ndarray, y: np.ndarray, theta: np.ndarray, alpha: float, num_iters: int) -> Tuple[np.ndarray, np.ndarray]:
    m = y.size
    J_history = np.zeros(shape=(num_iters, 1))

    for i in range(num_iters):
        predictions = X.dot(theta)
        theta_size = theta.size

        for it in range(theta_size):
            temp = X[:, it].reshape((m, 1))
            errors_x1 = (predictions - y) * temp
            theta[it][0] -= alpha * (1.0 / m) * errors_x1.sum()

        J_history[i, 0] = compute_cost(X, y, theta)

    return theta, J_history

data = np.loadtxt('data/spambase.data', delimiter=',')

X = data[:, :2]
y = data[:, 2]
m = y.size

y = y.reshape((m, 1))

x, mean_r, std_r = feature_normalize(X)

it = np.ones(shape=(m, 3))
it[:, 1:3] = x

iterations = 100
alpha = 0.01

theta = np.zeros(shape=(3, 1))
theta, J_history = gradient_descent(it, y, theta, alpha, iterations)

print(theta, J_history)

plt.plot(np.arange(iterations), J_history)
plt.xlabel('Iterations')
plt.ylabel('Cost Function')
plt.show()
