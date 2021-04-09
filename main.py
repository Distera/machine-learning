import pandas as pd
from sklearn.model_selection import train_test_split
import datetime as DT
import math

from lab1 import lab1
from lab2 import lab2
from lab3 import lab3
from lab4 import lab4_data_standardization, lab4_acceleration
from lab5 import lab5


def highlight_features_in_dayaFrame(data, name_column):
    unique_values = []
    new_data = {}
    rez_list = []

    for text in data[name_column]:
        words = text.split(';')
        rez_list.append(words)
        for word in words:
            if word not in unique_values:
                unique_values.append(word)

    new_data = {unique_value: [] for unique_value in unique_values}

    for words in rez_list:
        for unique_value in unique_values:
            new_data[unique_value].append(1 if unique_value in words else 0)

    df = pd.DataFrame(new_data)
    return df, unique_values


def assign_number_to_value(data):
    count = 0
    new_data = {}
    unique_values = []
    rez_list = []

    for text in data:
        if text not in unique_values:
            unique_values.append(text)
            new_data[text] = count
            count = count + 1

    for text in data:
        rez_list.append(new_data[text])
    return rez_list


def get_steam_dataset():
    y_name = 'positive_ratings_percentage'
    data = pd.read_csv('steam.csv', sep=",")
    data['positive_ratings_percentage'] = data.apply(
        lambda row: row.positive_ratings / (row.negative_ratings + row.positive_ratings), axis=1)

    features = ['achievements', 'average_playtime', 'price']
    x = data.loc[:, features]
    y = (data.loc[:, ['positive_ratings_percentage']] * 100).astype(int)

    df2, unique_values = highlight_features_in_dayaFrame(data, 'categories')
    x = pd.concat([x, df2], axis=1, join="inner")

    x['release_date'] = [DT.datetime.strptime(d, '%Y-%m-%d').date().year for d in data['release_date']]
    x['publisher'] = assign_number_to_value(data['publisher'])
    x['platforms'] = assign_number_to_value(data['platforms'])
    x['owners'] = assign_number_to_value(data['owners'])
    return x, y, y_name


def get_healthcare_dataset():
    y_name = 'stroke'
    data = pd.read_csv('healthcare-dataset-stroke-data.csv', sep=",")
    features = ['avg_glucose_level', 'age', 'hypertension', 'heart_disease']
    x = data.loc[:, features]
    y = data.loc[:, [y_name]]
    x['gender'] = assign_number_to_value(data['gender'])
    x['ever_married'] = assign_number_to_value(data['ever_married'])
    x['work_type'] = assign_number_to_value(data['work_type'])
    x['Residence_type'] = assign_number_to_value(data['Residence_type'])
    x['smoking_status'] = assign_number_to_value(data['smoking_status'])
    list_bmi = []
    for bmi in data['bmi']:
        list_bmi.append(29 if math.isnan(bmi) else bmi)
    data['bmi'] = list_bmi

    return x, y, y_name


if __name__ == '__main__':
    pd.options.display.max_columns = 20
    x, y, name_the_desired_value = get_healthcare_dataset()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=1)  # 80/20

    # lab1(x)
    # lab2(x, y[name_the_desired_value], name_the_desired_value)
    # lab3(x, y, name_the_desired_value)
    # lab4_data_standardization(x, y, name_the_desired_value, ["у пациента не было инсульта", "у пациента был инсульт"])
    # lab4_acceleration(x_train, x_test)
    # lab5(x_train, x_test, y_train, y_test)
