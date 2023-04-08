import gzip

import tensorflow as tf
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

    model = "C:/Users/picul/OneDrive/Documenti/RiverData/Irismodel.npz"

    percs = [0.4, 0.2, 0.4]
    for perc in percs:
        print(math.floor(100*perc))
