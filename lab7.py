from tensorflow import keras
from tensorflow.keras import layers
import tensorflow as tf
import math

from plot_diagram import plot_projected_and_expected


def lab7(x_train, x_test, y_train, y_test):
    y_pred = []
    inputs = keras.Input(shape=(9,), name='digits')
    x = layers.Dense(25, activation='relu', name='dense_1')(inputs)
    x = layers.Dense(25, activation='relu', name='dense_2')(x)
    outputs = layers.Dense(1, activation='sigmoid', name='predictions')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)

    # Укажем конфигурацию обучения (оптимизатор, функция потерь, метрики)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Обучим модель
    history = model.fit(x_train, y_train, epochs=10)#steps_per_epoch=200

    # значений потерь и метрик во время обучения
    print('\nhistory dict:', history.history)

    # Оценим модель на тестовых данных, используя "evaluate"
    print('\n# Оцениваем на тестовых данных')
    scores = model.evaluate(x_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))

    # Генерируем прогнозы на основе тестовой выборке
    for i in model.predict(x_test):
        (y_pred.append(1) if i >= 0.5 else y_pred.append(0))

    print(y_pred)
    plot_projected_and_expected(x_test, y_test, y_pred)
