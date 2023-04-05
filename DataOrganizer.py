import csv  # CSV data from River

from Errors import*      # Errors

DataFolderPath = 'C:/Users/picul/OneDrive/Documenti/RiverData/'
CSVRiverPath = 'sesia-scopello-scopetta.csv'


class DataOrganizer:
    def __init__(self, data_path):
        self.data_path = data_path
        self.data = None
        self.extracted_data = None

    def open_data(self):
        try:
            csv_file = open(self.data_path)
            self.data = csv.reader(csv_file, dialect='excel')
        except Exception as BroadError:
            raise IncorrectDataFile(self.data_path)

    def extract_data(self):
        self.extracted_data = [row for row in self.data]

    def print_data(self):
        print(self.data)

    def print_extracted_data(self):
        print(self.extracted_data)


    def get_data_numeric(self, element):
        element = self.extracted_data[3]
        print(element)
        date, hour_value = element[0].split(';')

        return element


class DataFormatReader:
    def __init__(self, format_path):
        self.format_path = format_path
        try:
            self.data = open(format_path, "r")
        except Exception:
            raise IncorrectFormatFile(format_path)
        self.rows = None
        self.data_types = {}
        self.input_files = []

    def create_data(self):
        self.rows = self.data.readlines()

    def parse_part_one(self):
        NotImplemented

    def parse_part_two(self):
        NotImplemented

    def act(self):
        NotImplemented
        Diomaledetto()

    def print_data(self):
        print(self.rows)


if __name__ == '__main__':

    Parse_data = 'C:/Users/picul/OneDrive/Documenti/RiverDataOrganizer.txt'

    DataOrg = DataOrganizer(DataFolderPath+CSVRiverPath)
    dataFormat = DataFormatReader(Parse_data)

    dataFormat.create_data()
    dataFormat.print_data()

