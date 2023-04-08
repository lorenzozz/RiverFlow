import numpy
import numpy as np

from DataOrganizer import DataFormatReader

__licence__ = 'Comune di Val Sesia, agenzia federale'
__version__ = '0.1'
__dependencies__ = ['numpy 1.20', 'tensorflow .', 'matplotlib .']

if __name__ == '__main__':

    make_file = "C:/Users/picul/OneDrive/Documenti/RiverData/Test.npz"
    with np.load(make_file) as model:
        print(model['Gianfranco'].shape)
        print(model['y'].shape)
        numpy.savez(make_file+'k')
    # Make_File = DataFormatReader(make_file)
    # Make_File.interpret()
