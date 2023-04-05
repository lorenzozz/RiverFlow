import csv  # CSV data from River
from Errors import *  # Errors

from re import findall, split  # Regex

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

        return element


class DataFormatReader:
    def __init__(self, format_path):

        self.format_path = format_path
        try:
            self.data = open(format_path, "r")
        except Exception:
            raise IncorrectFormatFile(format_path)
        self.rows = None

        # Note the difference. files_arglists is a dictionary where each key is a
        # file label, containing the complete argument list formatted for each file
        self.files_arglists = {}
        # On the other hand, variables is a dictionary containing the type of variable
        # that will be needed during construction of data row
        self.variables = {}
        self.input_files = {}
        self.formats = {}

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
                raise BadFormatStyle(self.format_path, "Bad pairing of declarations: " +
                                     "mismatched pair in declaration section ")

            # FILE<id> <Path> and arg_list {ID1}...{ID2} are paired, take even
            # and odd element of declaration section
            for file_token, arg_list in zip(decl_section[::2], decl_section[1::2]):

                source_line = str(self.rows.index(file_token))
                arg_list = arg_list.strip('\n')

                # Rule out malformed expressions
                if 'FILE' not in file_token:
                    raise BadFormatStyle(self.format_path, "Missing FILE token at line " +
                                         source_line)

                if file_token.count('<') + file_token.count('>') != 2:
                    raise MismatchedToken(self.format_path, "Incoherent use of delimiter tokens <> "
                                                            "or use of illegal character '>', '<' ")

                file_path = file_token.split('<')[1].split('>')[0].strip(' ')
                file_label = file_token.split('FILE')[1].split(':')[0].strip(' ')

                if not file_label or file_label == '':
                    raise BadFormatStyle(self.format_path, f"File label required for FILE {file_path}"
                                                           " at line " + source_line)

                # Add filepath file dictionary
                self.files_arglists[file_label] = arg_list
                self.input_files[file_label] = file_path

                if arg_list.count('{') != arg_list.count('}'):
                    raise MismatchedToken(self.format_path, "Mismatched '{' parenthesis in declaration "
                                                            "of variables at line " + source_line)

                split_arg_list = [a for arg in arg_list.split('{') for a in arg.split('}')]

                # Note that while labels occur between two occurrences of a '{'/'}' parenthesis, on
                # odd positions, format separators occur on even positions of the split_arg_list
                variable_names = split_arg_list[1::2]
                self.formats[file_label] = split_arg_list[::2]

                if any(i in self.variables for i in variable_names):
                    raise BadFormatStyle(self.format_path, f"Label already defined was redeclared"
                                                           " at line " + source_line)

                self.variables.update({name: None for name in variable_names})

        else:
            raise MissingSection(self.format_path, " Incorrect separation of declaration" +
                                 " section in format file ( missing newline \\n) ")

    def create_variable_vectors(self):
        NotImplemented

    def parse_part_two(self):

        decl_section_marker = self.rows.index('\n') + 1
        act_section_marker = self.rows[decl_section_marker + 1:].index('\n')

        if not act_section_marker:
            raise MissingSection(self.format_path, "Act segment not present" +
                                 "in format file ( missing newline \\n? ) ")

        recognized_data_types = ['Categorical', 'Numeric', 'Boolean']
        resolve_section = self.rows[decl_section_marker:act_section_marker]
        for declaration in resolve_section:
            source_line = str(self.rows.index(declaration))

            if ':' in declaration:

                variable_name = declaration.split(':')[0].strip()
                category = declaration.split(':')[1].strip()

                if category not in recognized_data_types:
                    raise BadFormatStyle(self.format_path, f"Unrecognized data category at "
                                                           f"line" + source_line)
                elif variable_name not in self.variables.keys():
                    raise BadFormatStyle(self.format_path, f"Unknown variable referenced at"
                                                           f" line" + source_line)

                self.variables[variable_name] = category

            elif '=' in declaration:

                new_var = declaration.split(':')[0].strip()
                reference = declaration.split(':')[1].strip()

                # No variable can be assigned before being declared and specified.
                if reference not in self.variables.keys() or not self.variables[reference]:
                    raise BadFormatStyle(self.format_path, f"Unknown or unspecified "
                                                           f"variable referenced"
                                                           f" at line" + source_line)
                else:
                    self.variables[new_var] = self.variables[reference]
        else:
            raise BadFormatStyle(self.format_path, f"Unrecognized token at line"
                                 + str(decl_section_marker))

        # Resolve section is also in charge of generating the vectors associated with each variable
        this.create_variable_vectors()

    def act(self):

        act_section_marker = self.rows[self.rows.index('\n') + 1:].index('\n')
        act_section = self.rows[act_section_marker:]

        if 'ACT:' not in act_section[0]:
            raise MissingSection(self.format_path, " Incorrect separation of acting" +
                                 " section in format file (missing ACT:?) ")

        for line in act_section:
            NotImplemented

    def print_data(self):
        print(self.rows)

    @staticmethod
    def parse_csv_line(format_string, line):
        data = []
        for sep in format_string:
            if sep:
                line = line.split(sep, 1)
                data.append(line[0])
                line = line[1]
        if not format_string[-1]:
            data.append(line)

        return data

    def parse_file(self, label):

        with open(self.input_files[label], "r") as csv_file:
            lines = csv.reader(csv_file, dialect='excel')
            if csv.Sniffer().has_header(csv_file.read(512)):
                lines.__next__()  # Glance over first line

            data = [self.parse_csv_line(self.formats[label], line[0]) for line in lines]

        return data


if __name__ == '__main__':
    Parse_data = 'C:/Users/picul/OneDrive/Documenti/RiverDataOrganizer.txt'

    DataOrg = DataOrganizer(DataFolderPath + CSVRiverPath)
    DataOrg.open_data()
    DataOrg.extract_data()

    dataFormat = DataFormatReader(Parse_data)

    dataFormat.create_data()
    dataFormat.print_data()
    dataFormat.parse_part_one()
    dataFormat.parse_part_two()
