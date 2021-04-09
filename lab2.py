import seaborn as sns
import matplotlib.pyplot as plt


def lab2(data, y_data, name_column):
    sns.violinplot(x=y_data)
    sns.displot(y_data)
    sns.boxplot(x=y_data)
    print(data.corr(), "\n------------------------\n")
    print(data.describe())

    X = data.copy(deep=True)
    X[name_column] = y_data
    sns.pairplot(X)
    plt.show()
