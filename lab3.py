import numpy as np
from matplotlib import pyplot as plt


def plot_linear_regression(x, y, y_pred, title, title_x, title_y):
    plt.figure(figsize=(12, 6))
    plt.plot(x, y_pred)
    plt.plot(x, y, 'ro')
    plt.title(title)
    plt.xlabel(title_x)
    plt.ylabel(title_y)
    plt.show()


def y_evaluation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    x_y_cov = (x - x_mean) * (y - y_mean)
    x_var = (x - x_mean) ** 2
    beta = x_y_cov.sum() / x_var.sum()
    alpha = y_mean - (beta * x_mean)
    y_pred = alpha + beta * x
    return y_pred


def lab3(X, y_data, y_column_name):
    data = X.copy(deep=True)
    data[y_column_name] = y_data
    for x_column_name in data:
        if x_column_name != y_column_name:
            y_pred = y_evaluation(data[x_column_name], data[y_column_name])
            plot_linear_regression(
                data[x_column_name],
                data[y_column_name],
                y_pred,
                y_column_name + " vs " + x_column_name,
                x_column_name,
                y_column_name
            )