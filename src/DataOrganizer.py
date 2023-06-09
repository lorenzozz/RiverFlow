import csv  # CSV data from River
import math  # Supported package
import Config

from matplotlib import pyplot as plt
from os import path

from Errors import *  # Errors
from VariableVectorAlgebra import *  # Vectorial algebra
from DatasetPlanner import *  # Make plans


class FileDeclGetter:
    # Static env
    glob_env = Config.__dict__

    @staticmethod
    def get_name_and_path(statement: str, file_tok: str):
        """ Get name and path of file declared according to standard syntax
        <file_tok> <file_name> = <file_path>
        File path can be any python expression that duck types to a string.
        Note that the only variables allowed inside a filepatch expression are those
        defined in the glob_env environment, e.g. the Config dictionary

        :param statement: The file declaration statement
        :param file_tok: The specific file token used in statement
        :return: both the file name and file path declared by user.
        """
        if file_tok not in statement:
            raise BadFormatStyle("", "Missing {f_tok} token inside statement {line}".format(
                f_tok=file_tok, line=statement))

        path_statement = statement.split('=')[1].strip()
        file_path = eval(path_statement, FileDeclGetter.glob_env)
        if not isinstance(file_path, str) and not hasattr(file_path, '__str__'):
            raise IncorrectFilePathExpr(file_path)

        file_label = statement.split(file_tok, 1)[1].split('=')[0].strip(' ')

        return file_label, file_path.__str__()


class DataFormatReader:
    def __init__(self, makefile_path):
        """  Initialize DataFormatReader object.
        Fills out format_path as input path, rows as the rows of
        the input file and various state variables to parse expressions
        in action section.
        :param: makefile_path: The path to the target makefile
        :return: None """
        self.format_path = makefile_path
        try:
            self.data = open(makefile_path, "r")
        except Exception:
            raise IncorrectFormatFile(makefile_path)
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
        """ Read data from format path and store lines into self.rows
        """
        self.rows = self.data.readlines()

        # Exclude blank lines and whole-line comments from interpretation
        self.rows = [
            r for r in self.rows
            if not str.isspace(r) and
               not r[0] == '#'
        ]

    def parse_part_one(self):
        """ Parse first part of format description file into a list of
        input files along with their corresponding format string
        :return: None
         """

        if '.decl\n' in self.rows:
            decl_section = [r for r in self.rows[1:self.rows.index('.res\n')]]
            if len(decl_section) % 2 != 0:
                raise BadFormatStyle(self.format_path,
                                     "Bad pairing of declarations: expected"
                                     "each declared file to have a format string, got a " +
                                     "mismatched pair in declaration section instead")

            # source_file <Fileid> =  <Path> and
            # {ID1}...{ID2} are paired, thus take even and odd element of declaration section
            for file_token, arg_list in zip(decl_section[::2], decl_section[1::2]):
                source_line = str(self.rows.index(file_token))
                arg_list = arg_list.strip('\n')

                file_label, file_path = FileDeclGetter.get_name_and_path(file_token, 'source_file')

                # Add filepath file dictionary
                self.input_files[file_label] = file_path

                # Check for trivial error in format string
                if arg_list.count('{') != arg_list.count('}'):
                    raise MismatchedToken(self.format_path, "Mismatched '{' parenthesis in declaration "
                                                            "of variables at line " + source_line)

                split_arg_list = [a for arg in arg_list.split('{') for a in arg.split('}')]

                # Note that while labels occur between two occurrences of a '{'/'}' parenthesis, on
                # odd positions, format separators occur on even positions of the split_arg_list
                variable_names = split_arg_list[1::2]

                self.formats[file_label] = split_arg_list[::2]
                self.files_arglists[file_label] = variable_names

                # No variable can be redeclared.
                if any(i in self.variables for i in variable_names):
                    raise BadFormatStyle(self.format_path,
                                         f"Label already defined was redeclared"
                                         " at line {}".format(source_line))

                self.variables.update({name: "N/A" for name in variable_names})

        else:
            raise MissingSection(self.format_path, "Incorrect separation of declaration" +
                                 " section in format file (missing .decl section?) ")

    def create_variable_vectors(self):
        """ Create vectors from specified variables.
        Failing to indicate a source file for a given variable causes
        an exception to be raised. No attempt to recovery is made
        if a filepath is malformed or problematic.
        """

        opened_files = {}
        for variable in self.variables:

            # Get owner file for each variable by searching for
            # the variable label inside each file's argument list
            owner = next(
                (label for label in self.input_files.keys()
                 if variable in self.files_arglists[label]),
                None
            )
            if not owner:
                raise BadFormatStyle(
                    self.format_path,
                    f"Expected variable {variable} to have an associated "
                    f"owner source file, got no file instead. Please specify a "
                    f"source for the variable."
                )

            if owner not in opened_files.keys():
                # Parse owner file
                opened_files[owner] = self.parse_file(
                    self.input_files[owner],
                    self.formats[owner]
                )

            # Note that Categorical and Boolean variables are not converted
            # to float automatically.
            as_type = VariableVectorManager.take_type(
                self.variables[variable],
                variable=variable
            )

            # Take index of target variable and file out of loop.
            var_col_i = self.files_arglists[owner].index(variable)
            file_data = opened_files[owner]

            self.var_vector.add_variable(variable, [
                row[var_col_i] for row in file_data
            ], as_type)

        return

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
                    raise BadFormatStyle(self.format_path, f"Unknown variable {variable_name}"
                                                           f" referenced at"
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
                plt.plot(x_data, y_data, label=x_var)
                plt.legend()
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
                file_name, file_path = FileDeclGetter.get_name_and_path(line, 'save_file')

                if '=' in line:
                    file_name = line.split('save_file')[1].split('=')[0].strip()
                else:
                    raise BadSaveFile(None, "Incorrect declaration of save file, missing '=' "
                                            "token at line " + line_number)

                save_files[file_name] = file_path
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

                # TODO:
                """
                TODO: Make module as a separate function
                Parameters : save_path, var_requested, format_list 
                
                with open(save_path) as new_csv_file:
                indices = [format_list.index(var) for var in var_requested]
                all_data_involved = [self.var_vector.get_variable(var) for var in var_requested]

                """

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

        plan_registered = dict()
        plan_save_files = {
            "None": None
        }
        log_save_files = dict()

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

                if 'plan' in statement:
                    file_label, file_path = FileDeclGetter.get_name_and_path(statement, 'plan_file')
                    plan_save_files[file_label] = file_path

                elif 'log' in statement:
                    file_label, file_path = FileDeclGetter.get_name_and_path(statement, 'log_file')
                    log_save_files[file_label] = file_path

            elif 'split' in statement:

                model_name = statement.split('split')[1].split('into')[0].strip()
                file_split = [f.strip() for f in statement.split('into')[1].split('as')[0].split(',')]
                percentages = [int(s) for s in statement.split('as')[1].split(',')]

                if sum(percentages) != 100:
                    raise BadSplitRequest(statement, "percentages don't sum up to 100.")

                plan_registered[model_name].change_field("split files", [plan_save_files[f] for f in file_split])
                plan_registered[model_name].change_field("proportions", percentages)

            # Out of compilation commands
            elif 'compile' in statement:

                plan_label = statement.split('compile')[1].split('into')[0].strip()

                # Only acceptable whenever another input has been
                # specified through splitting
                p_file_label = statement.split('into')[1].strip() if 'into' in statement else "None"

                plan_registered[plan_label].compile(plan_save_files[p_file_label])

            elif 'log ' in statement:
                plan_label = statement.split('log')[1].split('into')[0].strip()
                log_file = statement.split('into')[1].strip()
                plan_registered[plan_label].log(log_save_files[log_file])

            elif 'set ' in statement:
                set_value = statement.split('\"')[1].strip()
                model_name = next(i for i in statement.split(' ') if i in plan_registered.keys())
                field_name = statement.split(model_name)[1].split('=')[0].strip()
                # setting field_name to set_value in model_name
                plan_registered[model_name].change_field(field_name, set_value)

            elif not str.isspace(statement):
                raise BadAlignmentCommand(statement, ". Interrupting interpretation")

            current_row = current_row + 1

    @staticmethod
    def parse_file(inp_file: str, format_list: list, delim: str = ';'):
        """ Parses input file according to format list specified into format_list.

        The format list can be:
          - A list
          - A numpy array of unicode strings
          - An iterable
          - A string with a __next__ method
        Values inside inp_file are separate according to format_list according
        to each element inside the list.
        Each element of the format string is matched against each line in greedy
        search for a match. Whenever a match occurs, the input line is truncated
        according to the separator.

          Args:

          :param inp_file: The file path of the input file
          :param format_list: A format list used in the parsing operation
          :param delim: Il delimitatore fra gli elementi nel csv
          :return: The data fields as requested as a python iterable.

        """

        line_peek_amt = 2

        def _peek_lines(f):
            # Peek into file without incrementing file pointer
            # to not lose any data point.
            nonlocal line_peek_amt
            pos = f.tell()
            line = ''.join([f.readline() for _ in range(line_peek_amt)])
            f.seek(pos)
            return line

        if not path.exists(inp_file):
            raise FileNotFoundError("Attempting to parse a non existing file. Expected"
                                    "a valid filepath, found {}".__format__(inp_file))

        with open(inp_file, "r") as csv_file:
            lines = csv.reader(csv_file, dialect='excel', delimiter=delim)

            # Sniff over a few lines of target file, if it contains a header
            # glance over it
            if csv.Sniffer().has_header(_peek_lines(csv_file)):
                lines.__next__()

            def _parse_csv_line(format_string: list[str], line: str):
                parsed_data = []
                # Cut head of line
                if format_string[0]:
                    line = line.split(format_string[0], 1)[1]
                for sep in format_string[1:]:
                    # Sep might be empty string ""
                    if sep:
                        line = line.split(sep, 1)
                        parsed_data.append(line[0])
                        line = line[1]
                if not format_string[-1]:
                    parsed_data.append(line)
                return parsed_data

            data = [
                _parse_csv_line(format_list, delim.join(line))
                for line in lines
                if not str.isspace(delim.join(line)) and line is not None
            ]

        return data

    def interpret(self):
        """
        Go through the pipeline of interpreting all five section for requested make file
        :returns None
        """
        self.create_data()

        self.parse_part_one()  # .decl
        self.parse_part_two()  # .res
        self.act()  # .act
        self.parse_sap()  # .sap
        self.parse_make()  # .make


if __name__ == '__main__':
    # Parse_data = Config.URLROOT + r'\RiverData\NewIrisScript.txt'

    # Debug data, not present in production
    DataFolderPath = 'C:/Users/picul/OneDrive/Documenti/RiverData/'
    CSVRiverPath = 'sesia-scopello-scopetta.csv'

    Parse_data = Config.EXAMPLESROOT + "/River Height/Finaltest.makefile"
    dataFormat = DataFormatReader(Parse_data)
    dataFormat.interpret()
