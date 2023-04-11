import numpy
import numpy as np
import tensorflow as tf

from DataOrganizer import DataFormatReader

__licence__ = 'Comune di Val Sesia, agenzia federale'
__version__ = '0.1'
__dependencies__ = ['numpy 1.20', 'tensorflow .', 'matplotlib .']

if __name__ == '__main__':

    make_file = "C:/Users/picul/OneDrive/Documenti/RiverData/Test.npz"
    # with np.load(make_file) as model:
    #     print(model['Gianfranco'].shape)
    #     print(model['y'].shape)
    #     numpy.savez(make_file+'k')
    # Make_File = DataFormatReader(make_file)
    # Make_File.interpret()
    a= np.arange(0, 10)
    b = np.arange(2, 12)
    c = np.arange(5, 15)
    x = np.stack((a,b,c), axis=1)
    # print(len(x[0]))
    a = ..., 0
    print(a[0])
    a = np.arange(0,10)
    a = a[0]

    if a is not None and hasattr(a, '__len__'):
        print("1")

    if np.iterable(a):
        print("2")

    if hasattr(a, '__len__'):
        print("3")

    if np.size(a) > 0:
        print("4")

    NewModel = tf.keras.models.Sequential()
    # m = np.lib.stride_tricks.sliding_window_view(x, (2, 3))
    # print([np.concatenate(*w) for w in m])
