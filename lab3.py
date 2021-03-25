import numpy as np
from matplotlib import pyplot as plt

def Plot_linear_regression(x, y, ypred, titel, titel_x,  titel_y):
    plt.figure(figsize=(12, 6))
    plt.plot(x, ypred)
    plt.plot(x, y, 'ro')
    plt.title(titel)
    plt.xlabel(titel_x)
    plt.ylabel(titel_y)
    plt.show()

def Y_evaluation(x, y):
    xmean = np.mean(x)
    ymean = np.mean(y)
    xycov = (x - xmean) * (y - ymean)
    xvar = (x - xmean) ** 2
    beta = xycov.sum() / xvar.sum()
    alpha = ymean - (beta * xmean)
    ypred = alpha + beta * x
    return ypred

def lab3(data):

    x_value = 'positive_ratings_percentage'
    for y_value in data:
        if str(data[y_value].dtype) != 'object' and y_value != x_value:
            ypred = Y_evaluation(data[x_value], data[y_value])
            Plot_linear_regression(
                data[x_value],
                data[y_value],
                ypred,
                x_value+" vs "+ y_value,
                x_value,
                y_value
            )