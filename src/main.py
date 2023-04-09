import numpy
import numpy as np

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
    x = np.vstack((a,b,c))
    print(x)
    m = np.lib.stride_tricks.sliding_window_view(x, 3, axis=1)
    print(m)