import csv  # CSV data from River
from Errors import *  # Errors

# Debug data, not present in production
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
        except Exception:
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
        """
        Read data from format path and store lines into self.Rows
        :return: None
        """
        self.rows = self.data.readlines()

    def parse_part_one(self):
        """ Parse first part of format description file into a list of input files along
        with their corresponding format string
        :return: None
         """

        # Note: as format is required to have a blank newline before start of
        # Part 2, we can find the index of the empty element '\n' inside self.rows
        # to find where the declaration section ends

        if '\n' in self.rows:
            decl_section = self.rows[:self.rows.index('\n')]
            if len(decl_section) % 2:
                raise BadFormatStyle(self.format_path, " Bad pairing of declarations: " +
                                     "mismatched pair in declaration section ")

            # FILE<id> <Path> and arg_list {ID1}...{ID2} are paired, take even
            # and odd element of declaration section
            for file_token, arg_list in zip(decl_section[::2], decl_section[1::2]):

                # Rule out malformed expressions
                if 'FILE' not in file_token:
                    raise BadFormatStyle(self.format_path, " Missing FILE token at line " +
                                         str(self.rows.index(file_token)))

                if file_token.count('<') + file_token.count('>') != 2:
                    raise BadFormatStyle(self.format_path, " Incoherent use of delimiter tokens <> " +
                                         "or use of illegal character '>', '<' ")

                file_path = file_token.split('<')[1].split('>')[0].strip(' ')
                file_label = file_token.split('FILE')[1].split(':')[0].strip(' ')

                print(file_path)
                print(file_label)
        else:
            raise BadFormatStyle(self.format_path, " Incorrect separation of declaration" +
                                 " section in format file ( missing newline \\n) ")

    def parse_part_two(self):
        NotImplemented

    def act(self):
        NotImplemented

    def print_data(self):
        print(self.rows)


if __name__ == '__main__':
    Parse_data = 'C:/Users/picul/OneDrive/Documenti/RiverDataOrganizer.txt'

    DataOrg = DataOrganizer(DataFolderPath + CSVRiverPath)
    dataFormat = DataFormatReader(Parse_data)

    dataFormat.create_data()
    dataFormat.print_data()
    dataFormat.parse_part_one()