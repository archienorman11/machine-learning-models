import os
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt

def load_dataset(filename: str) -> pd.DataFrame:
    print(f"Loading dataset from file {filename}\n")
    data = pd.read_csv(filename, header=None, dtype=float)
    return zscore(data)

def zscore(data: pd.DataFrame) -> pd.DataFrame:
    return data.apply(lambda x: (x - x.mean()) / x.std(ddof=0), axis=0)

def prepare_data(data: pd.DataFrame) -> tuple:
    data.insert(0, "ones", 1)
    columns = [f"col_{i}" for i in range(58)]
    columns.append("y")
    data.columns = columns
    predictions = np.zeros(len(data))
    theta = np.zeros((len(data.columns) - 1, 10))
    return data, columns, predictions, theta

def logit_func(x: float) -> float:
    return 1 / (1 + np.exp(-x))

def update_theta(type: str, theta: np.ndarray, fold: int, alpha: float, X: np.ndarray, y: np.ndarray, train: np.ndarray) -> np.ndarray:
    if type == 'linear':
        update = alpha / len(train) * np.dot((y - np.dot(X, theta[:, fold])), X)
    elif type == 'logistic':
        update = alpha / len(train) * np.dot((y - logit_func(np.dot(X, theta[:, fold]))), X)
    else:
        raise ValueError("Invalid type")

    return theta[:, fold] + update

def cost_function(type: str, theta: np.ndarray, fold: int, validation: np.ndarray, index: np.ndarray) -> float:
    correct_y = validation[:, -1]
    correct_x = validation[:, :-1]
    predicted_y = np.dot(correct_x, theta[:, fold])

    if type == 'linear':
        mse = np.sum(np.square(predicted_y - correct_y)) / (2 * len(correct_y))
    elif type == 'logistic':
        mse = np.sum(np.square(logit_func(predicted_y) - correct_y)) / (2 * len(correct_y))
    else:
        raise ValueError("Invalid type")

    return mse

def gradient_descent(data: pd.DataFrame, theta: np.ndarray, alpha: float, tolerance: float, type: str, convergence: str) -> tuple:
    old_mean_error = 10
    avg_mean_error = 5
    max_iters = 700
    epoch = 0
    avg_mse_list = []

    while (old_mean_error - avg_mean_error) > tolerance and epoch <= max_iters:
        sq_means_list = []

        for fold in range(10):
            index = np.arange(fold, len(data), 10)
            validation = data.loc[index].values
            train = data.loc[~data.index.isin(index)].values

            if convergence == 'stochastic':
                for item in range(len(train)):
                    X = train[item, :-1]
                    y = train[item, -1]
                    theta[:, fold] = update_theta(type, theta, fold, alpha, X, y, train)

            elif convergence == 'batch':
                X = train[:, :-1]
                y = train[:, -1]
                theta[:, fold] = update_theta(type, theta, fold, alpha, X, y, train)

            else:
                raise ValueError("Invalid convergence")

            mse = cost_function(type, theta, fold, validation, index)
            sq_means_list.append(mse)

        old_mean_error = avg_mean_error
        avg_mean_error = sum(sq_means_list) / len(sq_means_list)
        avg_mse_list.append((epoch, avg_mean_error))

        print(f"Ending epoch {epoch} with average mean squared error of {avg_mean_error} ({type} and {convergence})")

        epoch += 1

    return avg_mse_list, epoch


def print_results(epoch: int, values: list) -> None:
    _, ax = plt.subplots()
    ax.plot(range(epoch), values)
    ax.set(xlabel='Iterations', ylabel='Cost Function')
    plt.show()


def main() -> None:
    data = load_dataset("data/spambase.data")
    data, columns, predictions, theta = prepare_data(data)

    lin_stoch1, iters = gradient_descent(data, theta, 0.1, 0.000001, 'linear', 'stochastic')
    epoch, values = lin_stoch1[0], lin_stoch1[1]

    gradient_descent(data, theta, 0.075, 0.0000001, 'linear', 'stochastic')
    gradient_descent(data, theta, 0.075, 0.0000001, 'linear', 'batch')

    gradient_descent(data, theta, 0.075, 0.0000001, 'logistic', 'stochastic')
    gradient_descent(data, theta, 0.2, 0.0000001, 'logistic', 'stochastic')

    gradient_descent(data, theta, 0.075, 0.0000001, 'logistic', 'batch')
    gradient_descent(data, theta, 0.2, 0.0000001, 'logistic', 'batch')

    # Uncomment the line below to display the plot
    # print_results(epoch, values)


if __name__ == '__main__':
    main()

