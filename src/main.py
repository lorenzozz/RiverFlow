import numpy as np
import Config

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
        print(model['y'][0])
