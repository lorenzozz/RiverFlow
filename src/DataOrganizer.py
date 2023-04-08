import csv  # CSV data from River
import math  # Supported package
from matplotlib import pyplot as plt

from Errors import *  # Errors
from VariableVectorAlgebra import *  # Vectorial algebra
from DatasetPlanner import *  # Make plans


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


class DataFormatReader:
    def __init__(self, format_path):
        """  Initialize DataFormatReader object.
        Fills out format_path as input path, rows as the rows of
        the input file and various state variables to parse expressions
        in action section.
        :return: None """
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
        # Input_files contains labels and filepath of each input file
        self.input_files = {}
        # Formats contains a copy of a file format disposition when
        # it is necessary to parse subexpressions
        self.formats = {}

        self.var_vector = VariableVectorManager()

    def create_data(self):
        """ Read data from format path and store lines into self.Rows
        :return: None """
        self.rows = self.data.readlines()
        # Exclude blank lines and whole-line comments from interpretation
        self.rows = [r for r in self.rows if not str.isspace(r) and not r[0] == '#']

    def parse_part_one(self):
        """ Parse first part of format description file into a list of input files along
        with their corresponding format string
        :return: None """

        # Note: as format is required to have a blank newline before start of
        # Part 2, we can find the index of the empty element '\n' inside self.rows
        # to find where the declaration section ends

        if '.decl\n' in self.rows:
            decl_section = [r for r in self.rows[1:self.rows.index('.res\n')] if r != '\n']
            if len(decl_section) % 2 != 0:
                raise BadFormatStyle(self.format_path, "Bad pairing of declarations: " +
                                     "mismatched pair in declaration section ")

            # file<id> <Path> and arg_list {ID1}...{ID2} are paired, take even
            # and odd element of declaration section
            for file_token, arg_list in zip(decl_section[::2], decl_section[1::2]):
                source_line = str(self.rows.index(file_token))
                arg_list = arg_list.strip('\n')

                # Rule out malformed expressions
                if 'source_file' not in file_token:
                    raise BadFormatStyle(self.format_path, "Missing source_file token at line " +
                                         source_line)

                if file_token.count('\"') != 2:
                    raise MismatchedToken(self.format_path, "Incoherent use of token '=' "
                                                            "or use of illegal character \"")

                file_path = file_token.split('\"')[1].strip()
                file_label = file_token.split('source_file', 1)[1].split('=')[0].strip(' ')
                if not file_label or file_label == '':
                    raise BadFormatStyle(self.format_path, f"File label required for source_file {file_path}"
                                                           " at line " + source_line)

                # Add filepath file dictionary
                self.input_files[file_label] = file_path

                if arg_list.count('{') != arg_list.count('}'):
                    raise MismatchedToken(self.format_path, "Mismatched '{' parenthesis in declaration "
                                                            "of variables at line " + source_line)

                split_arg_list = [a for arg in arg_list.split('{') for a in arg.split('}')]

                # Note that while labels occur between two occurrences of a '{'/'}' parenthesis, on
                # odd positions, format separators occur on even positions of the split_arg_list
                variable_names = split_arg_list[1::2]

                self.formats[file_label] = split_arg_list[::2]
                self.files_arglists[file_label] = variable_names

                if any(i in self.variables for i in variable_names):
                    raise BadFormatStyle(self.format_path, f"Label already defined was redeclared"
                                                           " at line " + source_line)

                self.variables.update({name: None for name in variable_names})

        else:
            raise MissingSection(self.format_path, "Incorrect separation of declaration" +
                                 " section in format file (missing .decl?) ")

    def create_variable_vectors(self):

        opened_files = {}
        for variable in self.variables:

            owner_file = next((label for label in self.input_files.keys()
                               if variable in self.files_arglists[label]), None)

            if not owner_file:
                raise BadFormatStyle(self.format_path, f"Variable {variable} has no" +
                                     f" associated source file redeclared at line ")

            if owner_file not in opened_files.keys():
                opened_files[owner_file] = self.parse_file(owner_file)

            var_col_i = self.files_arglists[owner_file].index(variable)

            # Pass new variable onto vector mathematics manager
            file_data = opened_files[owner_file]
            # Categorical and Boolean variables are not converted to float autonomously.
            as_type = self.var_vector.take_type(self.variables[variable])
            self.var_vector.add_variable(variable, [
                row[var_col_i] for row in file_data
            ], as_type)

    def parse_part_two(self):

        res_section_marker = self.rows.index('.res\n') + 1
        act_section_marker = self.rows.index('.act\n')

        if not act_section_marker:
            raise MissingSection(self.format_path, "Act segment not present" +
                                 "in format file ( missing newline \\n? ) ")

        recognized_data_types = ['categorical', 'numeric', 'boolean', 'integer']
        resolve_section = self.rows[res_section_marker:act_section_marker]

        copy_action: list[tuple] = []

        for declaration in [r for r in resolve_section if not str.isspace(r)]:

            if '#' in declaration:
                declaration = declaration.split('#')[0]

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

                new_var = declaration.split('=')[0].strip()
                reference = declaration.split('=')[1].strip()

                # No variable can be assigned before being declared and specified.
                if reference not in self.variables.keys() or not self.variables[reference]:
                    raise BadFormatStyle(self.format_path, f"Unknown or unspecified "
                                                           f"variable referenced"
                                                           f" at line" + source_line)
                else:
                    copy_action.append((new_var, reference))
            else:
                raise BadFormatStyle(self.format_path, f"Unrecognized token at line "
                                     + source_line)

        # Resolve section is also in charge of generating the vectors associated
        # with each variable as it must remember assignments and do deep-copy of
        # numpy vectors associated with copied variables
        self.create_variable_vectors()

        for new, copied in copy_action:
            self.var_vector.add_copy_of(new, copied)
            self.variables[new] = self.variables[copied]

    def act(self, no_except=False):

        act_sec = self.rows.index('.act\n')
        sap_sec = self.rows.index('.sap\n')

        act_section = self.rows[act_sec:sap_sec]

        if '.act' not in act_section[0]:
            raise MissingSection(self.format_path, " Incorrect separation of acting" +
                                 " section in format file (missing .act?) ")

        self.var_vector.load_grammar_mapper()
        self.var_vector.change_error_mode(no_except)

        recognized_packages = {
            "numpy": np,
            "math": math,
            "datetime": datetime
        }

        for line in [li for li in act_section[1:] if li != '\n']:

            line_number = self.rows.index(line)

            if 'new ' in line:
                # A new variable has been declared. Note that variables
                # initialized in this manner must be assigned a value before
                # use.
                # Note that execution will not continue if VariableVectorAlgebra
                # manager throws an exception over an impossible vectorial operation.
                var_name = line.split('new ')[1].strip() if '=' not in line else \
                    line.split('new ')[1].split('=')[0].strip()

                if not var_name:
                    raise BadFormatStyle(self.format_path, f"Incorrect data declaration"
                                         + " in action segment at line " + str(line_number))

                # Variable has no data to be initialized with, hence it cannot be used
                # as rhs.
                self.variables[var_name] = None
                self.var_vector.add_variable(var_name, [], self.var_vector.take_type("numeric"))

                # Support for immediate initialization.
                if '=' in line:
                    self.var_vector.execute_line(line.split('new')[1], line_number)
            elif 'import' in line:
                package_name = line.strip().split(' ')[1]
                if package_name in recognized_packages.keys():

                    # Support for package aliases
                    if 'as' in line:
                        alias = line.strip().split(' ')[3]
                    else:
                        alias = package_name

                    self.var_vector.add_package(alias, recognized_packages[package_name])

                else:
                    raise UnrecognizedPackageReference(f"Package {package_name} not recognized.")
            elif 'plot' in line:
                if 'against' in line:
                    y_var = line.split('plot')[1].split('against')[0].strip()
                    x_var = line.split('against')[1].strip()

                    x_data = self.var_vector.get_variable(x_var)
                    y_data = self.var_vector.get_variable(y_var)
                else:
                    y_var = line.split('plot')[1].strip()
                    x_var = y_var

                    x_data = np.arange(0, self.var_vector.get_sizeof_var(y_var))
                    y_data = self.var_vector.get_variable(y_var)

                if self.var_vector.get_sizeof_var(x_var) != self.var_vector.get_sizeof_var(y_var):
                    raise IncorrectPlotRequest(x_var, y_var, "size do not match.")

                plt.figure()
                plt.plot(x_data, y_data)
                plt.xlabel = x_var
                plt.ylabel = y_var
                plt.show()

            else:
                self.var_vector.execute_line(line, line_number)

    def parse_sap(self):

        save_files = {}
        sap_sec = self.rows.index('.sap\n')
        make_sec = self.rows.index('.make\n')

        for line in [li for li in self.rows[sap_sec + 1:make_sec] if li != '\n']:
            line_number = str(self.rows.index(line))

            # Do not parse comments.
            line = line.split('#')[0] if '#' in line else line

            if 'save_file' in line:
                if '=' in line:
                    file_name = line.split('save_file')[1].split('=')[0].strip()
                else:
                    raise BadSaveFile(None, "Incorrect declaration of save file, missing '=' "
                                            "token at line " + line_number)

                save_files[file_name] = line.split('\"')[1].strip()
            elif 'save ' in line:
                # Save requested variables with indicate format. a ';' signals csv
                # to separate into different columns, the rest of the characters are taken
                # literally and appended to the data value.
                if 'into' not in line:
                    raise BadSaveFile(None, "File label not specified while saving at"
                                            " line " + line_number)

                format_expression = line.split('\"')[1].split('\"')[0]
                file_requested = line.split('into')[1].strip()

                if file_requested not in save_files.keys():
                    raise BadSaveFile(None, "save_file label not recognized at"
                                            " line " + line_number)

                # Attempting to write a save file with variables of different
                # dimension is nonsense. Find any such occurrence and issue an
                # error
                total = [el for sl in format_expression.split('}') for el in sl.split('{')]
                var_requested = total[1::2]

                var_dims = [self.var_vector.variables_dims[var] for var in var_requested]

                if var_dims.count(var_dims[0]) != len(var_dims):
                    raise IncompatibleVecSizes(file_requested, "Cannot perform" +
                                               " operations with vectors of different sizes at line" + line_number)

                with open(save_files[file_requested], "w") as new_csv_file:
                    indices = [total.index(var) for var in var_requested]
                    all_data_involved = [self.var_vector.get_variable(var) for var in var_requested]

                    lines = []
                    for pack in zip(*all_data_involved):
                        for i, val in zip(indices, pack):
                            total[i] = str(val)
                        replace = ''.join(total) + '\n'
                        lines.append(replace)

                    new_csv_file.writelines(lines)

    def parse_make(self):

        plan_sec = self.rows.index('.make\n') + 1
        current_row = plan_sec

        plan_registered = {}
        plan_save_files = {}
        log_save_files = {}

        while current_row < len(self.rows):
            statement = self.rows[current_row]
            if 'begin plan' in statement:
                plan_name = statement.split('begin plan')[1]
                if 'expecting' in plan_name:
                    plan_name = plan_name.split('expecting')[0]
                plan_name = plan_name.strip()

                # Delegate the building of the actual plan to a DatasetPlanner Object,
                # then find the end of the declaration and proceed in parsing simpler
                # statements.
                plan_registered[plan_name] = DatasetPlanner(self.rows[current_row:], self.var_vector, plan_name)
                plan_registered[plan_name].parse()
                current_row = next(i for i in range(current_row, len(self.rows)) if 'end plan' in self.rows[i])
            # House-keeping file management commands
            elif '_file ' in statement:

                file_path = statement.split('\"')[1].strip()
                if 'plan' in statement:
                    file_label = statement.split('plan_file ')[1].split('=')[0].strip()
                    plan_save_files[file_label] = file_path
                elif 'log' in statement:
                    log_label = statement.split('log_file ')[1].split('=')[0].strip()
                    log_save_files[log_label] = file_path

            # Out of compilation commands
            elif 'compile' in statement:

                plan_label = statement.split('compile')[1].split('into')[0].strip()
                plan_file_label = statement.split('into')[1].strip()
                plan_registered[plan_label].compile(plan_save_files[plan_file_label])

            elif 'log ' in statement:
                plan_label = statement.split('log')[1].split('into')[0].strip()
                log_file = statement.split('into')[1].strip()
                plan_registered[plan_label].log(log_save_files[log_file])

            elif 'set' in statement:
                set_value = statement.split('\"')[1].strip()
                model_name = next(i for i in statement.split(' ') if i in plan_registered.keys())
                field_name = statement.split(model_name)[1].split('=')[0].strip()
                # setting field_name to set_value in model_name
                plan_registered[model_name].change_field(field_name, set_value)

            elif not str.isspace(statement):
                raise BadAlignmentCommand(statement, ". Interrupting interpretation")

            current_row = current_row + 1

    def print_data(self):
        print(self.rows)

    @staticmethod
    def parse_csv_line(format_string, line):
        data = []
        # Cut head of line
        # print(line, format_string)

        if format_string[0]:
            line = line.split(format_string[0], 1)[1]
        for sep in format_string[1:]:
            if sep:
                line = line.split(sep, 1)
                data.append(line[0])
                line = line[1]
        if not format_string[-1]:
            data.append(line)
        return data

    def parse_file(self, label):

        with open(self.input_files[label], "r") as csv_file:
            lines = csv.reader(csv_file, dialect='excel', delimiter=';')

            if csv.Sniffer().has_header(csv_file.readline()):
                lines.__next__()  # Glance over first\ line

            data = [self.parse_csv_line(self.formats[label], ';'.join(line)) for line in lines
                    if not str.isspace(';'.join(line)) and line]

        return data

    def interpret(self):
        """
        Go through the pipeline of interpreting all five section for requested make file
        :returns None
        """
        self.create_data()
        # .decl
        self.parse_part_one()
        # .res
        self.parse_part_two()
        # .act
        self.act()
        # .sap
        self.parse_sap()
        # .make
        self.parse_make()


if __name__ == '__main__':
    Parse_data = 'C:/Users/picul/OneDrive/Documenti/past-riverdata.txt'
    Parse_datat = 'C:/Users/picul/OneDrive/Documenti/RiverData/NewIrisScript.txt'

    # Debug data, not present in production
    DataFolderPath = 'C:/Users/picul/OneDrive/Documenti/RiverData/'
    CSVRiverPath = 'sesia-scopello-scopetta.csv'

    dataFormat = DataFormatReader(Parse_datat)

    dataFormat.create_data()
    dataFormat.parse_part_one()
    dataFormat.parse_part_two()
    dataFormat.act()
    dataFormat.parse_sap()
    dataFormat.parse_make()
