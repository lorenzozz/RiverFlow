import random

import numpy as np
from Config import *
from DataOrganizer import DataFormatReader
import tensorflow as tf
from matplotlib import pyplot as plt


def _pad_missing(data_path: str, format_str: str):
    format_list = [k for elem in format_str.split('{') for k in elem.split('}')]
    parsed_data = DataFormatReader.parse_file(data_path, format_list[::2])

    train_data = [[float(el) for el in item[:-1]] for item in parsed_data]
    # Encode categorical values.
    num = {'Iris-setosa': -2, 'Iris-virginica': 2, 'Iris-versicolor': 4}
    for i in range(0, len(train_data)):
        train_data[i].append(
            num[parsed_data[i][-1]]
        )

    noise_samples = 12 * len(train_data)
    random_samples = 4*len(train_data)
    test_shit = train_data.copy()
    perfect_train = train_data.copy()

    def _gen_noisy():
        nonlocal train_data
        for _ in range(0, noise_samples):
            # Get zeroing out percentage
            p = random.randint(0, 10) / 10
            el = random.choice(test_shit)

            noise = [e if random.uniform(0, 1) > p else 0.0 for e in el]
            train_data.append(noise)
            perfect_train.append(el)
        for _ in range(0, random_samples):

            noise = np.random.randn(len(train_data[0]))
            el = random.choice(test_shit)

            train_data.append(list(np.array(el)+noise))
            perfect_train.append(el)
    _gen_noisy()

    zipped = list(zip(train_data, perfect_train))
    random.shuffle(zipped)
    train_data, perfect_train = zip(*zipped)
    train_data = list(train_data)
    perfect_train = list(perfect_train)
    print(perfect_train)
    layer_size = len(train_data[0])
    model = tf.keras.Sequential(
        [
            tf.keras.layers.InputLayer(input_shape=layer_size),
            tf.keras.layers.Dense(200, activation="relu"),
            tf.keras.layers.Dropout(rate=0.01),
            tf.keras.layers.Dense(150, activation="relu"),
            tf.keras.layers.Dropout(rate=0.01),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dropout(rate=0.01),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(100, activation="relu"),
            tf.keras.layers.Dense(layer_size),
        ]
    )
    train_ds = tf.data.Dataset.from_tensor_slices(
        (train_data, perfect_train)).shuffle(400).batch(32)

    model.summary()
    model.compile(optimizer='adam',
                  loss=tf.keras.losses.MeanSquaredError(),
                  metrics=['mean_squared_error'])

    model.fit(train_ds, epochs=400)
    model.evaluate(train_data)

    true_val = [el[0] for el in test_shit]
    result = model.predict(test_shit)
    pred_nonzero = [el[0] for el in result]

    for i in range(0, len(test_shit)):
        for j in range(0, len(test_shit[0])):
            if random.randint(1, 10) > 3:
                test_shit[i][j] = 0
    print(test_shit)
    result = model.predict(test_shit)
    what_got_in = [el[0] for el in test_shit]
    predicted_val = [el[0] for el in result]

    plt.plot(pred_nonzero, label='predicted with no zeros', linewidth=0.4)
    plt.plot(predicted_val, label='predicted with zeros', linewidth=0.6, color='magenta')
    plt.plot(true_val, label='reality', linewidth=0.4)
    # plt.plot(what_got_in, label='what got in', linewidth=0.4)
    plt.legend()
    plt.show()

def _knn_impute(data_path: str, format_str: str):



if __name__ == '__main__':
    _pad_missing(EXAMPLESROOT + '/Iris Dataset/Iris-dataset/iris.data',
                 '{a},{b},{c},{d},{name}')
