import matplotlib.pyplot as plt

def plot_projected_and_expected(x_test, y_test, predict):
    x = [number for number in range(y_test.count())]
    plt.scatter(x, y_test, color='red', alpha=0.3)
    plt.scatter(x, predict, color='blue', alpha=0.3)
    # plt.gca().set(xlim=(700, 800))
    plt.title('Прогноз y на тестовых данных')
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.show()