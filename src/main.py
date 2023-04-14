import numpy as np
import Config
import tensorflow as tf

__licence__ = 'Comune di Val Sesia, agenzia federale'
__version__ = '0.1'
__dependencies__ = ['numpy 1.20', 'tensorflow .', 'matplotlib .']

if __name__ == '__main__':
    None

    make_file = Config.EXAMPLESROOT + "/River Height/savefile.npz"
    with np.load(make_file) as model:
        print(model['x'][0], model['y'][0])
        print(model['x'][1], model['y'][1])

        print(model['x'].shape)
        print(np.size(model['y'][0]))

        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(2457, activation='relu'),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(168)
        ])
