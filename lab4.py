import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib import pyplot as plt


def lab4_data_standardization(x, y, name, name_colums):
    x = StandardScaler().fit_transform(x)
    print(x)

    # Проекция PCA в 2D
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data=principalComponents,
                               columns=['principal component 1', 'principal component 2']
                               )
    # print(principalDf)
    pca = PCA(.95)
    pca.fit(principalDf)
    finalDf = pd.concat([principalDf, y], axis=1)
    print(finalDf)

    # Визуализация 2д проекции
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(1, 1, 1)
    ax.set_xlabel('Principal Component 1', fontsize=15)
    ax.set_ylabel('Principal Component 2', fontsize=15)
    ax.set_title('2 component PCA', fontsize=20)
    targets = [name_colums[0], name_colums[1]]
    colors = ['r', 'g', 'b']
    indicesToKeep = finalDf[name] == 0
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c=colors[1]
               , alpha=0.5
               , s=50)
    indicesToKeep1 = finalDf[name] == 1
    ax.scatter(finalDf.loc[indicesToKeep1, 'principal component 1']
               , finalDf.loc[indicesToKeep1, 'principal component 2']
               , c=colors[0]
               , alpha=0.5
               , s=50)

    ax.legend(targets)
    ax.grid()
    plt.show()
    print(pca.explained_variance_ratio_)


def lab4_acceleration(train_data, test_data):
    # PCA для ускорения алгоритмов машинного обучения

    scaler = StandardScaler()  # нормализует объекты (mean = 0 и standard deviation = 1)
    scaler.fit(train_data)
    # Применить преобразование как к набору обучения, так и к набору тестов.
    train_img = scaler.transform(train_data)
    test_img = scaler.transform(test_data)
    # print(train_img)
    # print("--------------------------------------")
    # print(test_img)

    # Создание экземпляра модели (95% дисперсии)
    pca = PCA(.95)
    pca.fit(train_img)

    train_img = pca.transform(train_data)
    test_img = pca.transform(test_data)
    print("--------------------------------------")
    print(train_img)
    print("--------------------------------------")
    print(test_img)
