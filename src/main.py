import gzip

import numpy
import csv
import time
from LogManager import *
from Errors import *
from DataOrganizer import *

import numpy as np

if __name__ == '__main__':
    DataFolderPath = 'C:/Users/picul/OneDrive/Documenti/RiverData/'
    CSVRiverPath = 'sesia-scopello-scopetta22.csv'

    model = "C:/Users/picul/OneDrive/Documenti/RiverData/save_dataset.npz"
    model_data = numpy.load(model, allow_pickle=True)
    print(sorted(model_data))
