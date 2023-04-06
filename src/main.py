import numpy
import csv
from LogManager import*
from Errors import*
from DataOrganizer import*

import numpy as np


if __name__ == '__main__':
    DataFolderPath = 'C:/Users/picul/OneDrive/Documenti/RiverData/'
    CSVRiverPath = 'sesia-scopello-scopetta22.csv'
    with open(DataFolderPath+CSVRiverPath, "w") as csv_file:
        writer = csv.writer(csv_file)
