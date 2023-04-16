import numpy as np
import Config
import tensorflow as tf
import matplotlib.pyplot as plt

__licence__ = 'Comune di Val Sesia, agenzia federale'
__version__ = '0.1'
__dependencies__ = ['numpy 1.20', 'tensorflow .', 'matplotlib .']

if __name__ == '__main__':
    make_file = Config.EXAMPLESROOT + "/River Height/savefile.npz"
    test_file = Config.EXAMPLESROOT + "/River Height/test.npz"

    with np.load(make_file) as model_data:
        with np.load(test_file) as test_data:
            print(model_data['x'])
            print(test_data['x'])