import pandas as pd
from lab2 import lab2
from lab3 import lab3

if __name__ == '__main__':
    pd.options.display.max_columns = 20
    data = pd.read_csv('steam.csv', sep=",")
    data['positive_ratings_percentage'] = data.apply(
        lambda row: row.positive_ratings / (row.negative_ratings + row.positive_ratings), axis=1)

    lab2(data)
    lab3(data)
