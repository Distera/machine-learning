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


def lab3(data):
    x_value = 'positive_ratings_percentage'
    for y_value in data:
        if str(data[y_value].dtype) != 'object' and y_value != x_value:
            y_pred = y_evaluation(data[x_value], data[y_value])
            plot_linear_regression(
                data[x_value],
                data[y_value],
                y_pred,
                x_value + " vs " + y_value,
                x_value,
                y_value
            )
