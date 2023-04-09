import numpy

from DatasetErrors import *
from LogManager import *
from VariableVectorAlgebra import VariableVectorManager

import numpy as np  # Vector support, sliding view.
import time  # dates comparison


class Aligner:

    def __init__(self, var_vec, variables, aligning_variables):
        self.variables: list[str] = variables
        self.alignment: list[str] = aligning_variables

        self.windows = {}
        # Upper alignment budget and Lower alignment budget are stored in the same tuple
        self.budgets = {}
        self.var_vec: VariableVectorManager = var_vec
        self.init_align = None

        self.target_var = None
        # Initialize the aligning underlying structure
        # Necessary to map virtual indexes to logical indexes
        self.initial_align = {}

        self.create_alignment()

        # Describes whether a variable must generate n-dimensional views over its values or
        # single points
        self.window_generation_type = {}

    @staticmethod
    def get_formatted_date_equal(string):
        def is_date_equal(*dates):
            time_dates = [time.strptime(date, string) for date in [*dates]]
            return all(time_dates[0] == other_date for other_date in time_dates)

        return is_date_equal

    @staticmethod
    def index_equal(*numbers):
        all([numbers][0] == n_2 for n_2 in [numbers])

    def wants(self, var_name, request: list):

        # A variable's behaviour during model creation can only
        # be specified once per model.

        if var_name in self.window_generation_type.keys():
            raise VariableSliceRedefinition(var_name)
        self.window_generation_type[var_name] = request

        # Window keeping algorithm:
        # When a variable requests to keep a window to slide over the
        # data array, it must be checked whether the variable has the ability
        # to do so without influencing the behaviour of other variables.
        # example:

        # Data1: [...], Data1_alignment : [0, 1, 2, 3, 4, 5, 6, 7]
        # Data2: [...], Data2_alignment : [4, 5, 6, 7]
        # Alignment described:
        # 0   1   2   3   4   5   6   7       window: (0, 4) bottom_align = 0
        #     |---|---|---4   5   6   7       window: (2, 6) bottom_align = 2
        #                 ^ Begin of transcription

        # If data1 asks to keep a sliding window (e.g. 3) in the final model, there is no need
        # to change the behaviour of data2, for there are enough values of Data1 before
        # the beginning of the alignment to do so. Suppose now Data1 requests a window
        # of length 4. We are now obliged to throw away Data2 point 4, as it simply does not
        # have 4 values of data1 behind it.

        # 0   1   2   3   4   5   6   7       window: (0, 4) bottom_align = 0
        #     |---|---|---4---5   6   7       window: (2, 6) bottom_align = 2
        #                     ^ Begin of transcription

        budget = self.budgets[var_name]

        # TODO: REFactor
        # cant use for loop as lower window must grow while upper window must decrease
        # Lower window
        if request[0] > budget[0]:
            self.windows[var_name][0] += request[0] - budget[0]
            self.budgets[var_name][0] = 0
        else:
            self.budgets[var_name][0] -= request[0]

        # Upper window
        if request[1] > budget[1]:
            self.windows[var_name][1] -= request[1] - budget[1]
            self.budgets[var_name][1] = 0
        else:
            self.budgets[var_name][1] -= request[1]

    def show_status(self):
        print(self.windows)
        print(self.budgets)

    def add_target(self, target_var):
        self.target_var = target_var

    def get_convolution(self, var_name):
        """
        Returns an array containing all views over the data vector as requested from
        the description in the model. The return values are actually a view over the
        initial data vector and thus are read only.

        The routine computes the minimum upper bound and maximum lower bound
        from the window sizes kept in self.windows.
        It then computes the logical index of the usable slice of the aligned
        complete data plan by subtracting the initial align of the data.
        example:
                x
        0   1   2   3   4   5   6   7   8               D1
                2   3   4   5   6                       D2
                2   3   4   5   6   7   8               D3
        If a window size of 1 before a 2 forward is specified for D1, no
        change has to occur the alignment of other elements, as D1 has enough
        data points in the alignment to provide such a window.
            |---x---|---|
        0   1  ^2  ^3  ^4  ^5  ^6   7   8               D1
                2   3   4   5   6                       D2
                2   3   4   5   6   7   8               D3
        Max_upper = 6
        Min_lower = 2

        But  if D2 request a forward window of 2, the minimum upper window
        has to go down by 2 for each sample to have the same amount of data points.
            |---x---|---|
        0   1  ^2  ^3  ^4  ^5  ^6   7   8               D1
               ^2  ^3  ^4   5   6                       D2
                |---|---|
                2   3   4   5   6   7   8               D3
        Max_upper = 4
        Min_lower = 2

        This same process is applied for every request. The final maximum lower bound
        and minimum upper bound are the one used as specified above.

        The actual convolution is the computed over by np.lib.stride_tricks
        using the window size specified by the user file, as such
        window_size = 1+lower_window+upper_window

        :param var_name: the data vector
        :return: views over the array as indicated
        """
        max_lb = max(self.lower_bound)
        min_ub = min(self.upper_bound)

        l_bot = self.get_logical(max_lb - self.bot_slide(var_name), var_name)

        # Add 1 to account for python indexing
        l_top = 1 + self.get_logical(min_ub + self.top_slide(var_name), var_name)

        # The +1 accounts for the fact that the current element is included in the window
        # Thus the window has size (prev_request, 1+after_request)
        slider = self.top_slide(var_name) + self.bot_slide(var_name) + 1

        # Trim data up to usable window in order to slide window over it and collect n samples
        trimmed_data = self.var_vec.get_variable(var_name)[l_bot:l_top]

        conv_data = np.lib.stride_tricks.sliding_window_view(trimmed_data, slider)

        return conv_data

    def singleton(self, var_name):
        if var_name in self.window_generation_type.keys():
            raise VariableSliceRedefinition(var_name)
        self.window_generation_type[var_name] = [0, 0]

    @property
    def lower_bound(self):
        return [self.windows[var][0] for var in self.variables]

    @property
    def upper_bound(self):
        return [self.windows[var][1] for var in self.variables]

    def bot_slide(self, name):
        if isinstance(name, list):
            return [self.window_generation_type[var][0] for var in name]
        else:
            return self.window_generation_type[name][0]

    def top_slide(self, name):
        if isinstance(name, list):
            return [self.window_generation_type[var][1] for var in name]
        else:
            return self.window_generation_type[name][1]

    def get_logical(self, index, name):
        return index - self.initial_align[name]

    def create_alignment(self):

        # Find first element present in all aligning elements.
        var_data = [self.var_vec.get_variable(var) for var in self.alignment]
        intersection = var_data[0]
        for data in var_data:
            intersection = np.intersect1d(data, intersection)

        # Get bottom alignment, a mapped value describing how different
        # vectors map onto each other in alignment

        # example:
        # Data1: [...], Data1_alignment : [0, 1, 2, 3, 4]
        # Data2: [...], Data2_alignment : [2, 3, 4, 5, 6]
        # Alignment described:
        # 0   1   2   3   4                 window: (0, 4) bottom_align = 0
        #         2   3   4   5   6         window: (2, 6) bottom_align = 2
        bottom_aligns = [np.nonzero(data1 == intersection[0])[0][0] for data1 in var_data]
        self.init_align = max(bottom_aligns)
        bottom_aligns = [self.init_align - el for el in bottom_aligns]

        # Get top alignment, a mapped value describing the ceiling
        # of the alignment

        # example:
        # Data1: [...], Data1_alignment : [0, 1, 2, 3, 4]
        # Data2: [...], Data2_alignment : [2, 3, 4, 5, 6]
        # Alignment described:
        # 0   1   2   3   4                 window: (0, 4) top_align = 4
        #         2   3   4   5   6         window: (2, 6) bottom_align = 6

        sizes = [np.size(data) for data in var_data]
        up_locs = [size + bottom - 1 for size, bottom in zip(sizes, bottom_aligns)]
        min_up = min(up_locs)

        # Create windows and budgets for each data source to use in
        # windowing algorithm

        # example:
        # Data1: [...], Data1_alignment : [0, 1, 2, 3, 4]
        # Data2: [...], Data2_alignment : [2, 3, 4, 5, 6]
        # Alignment described:
        # 0   1   2   3   4                 window: (0, 4)
        #         2   3   4   5   6         window: (2, 6)
        #
        # Budgets are computed as follows
        # INIT_ALIGN_BOTTOM = max(bottom-aligns)
        # INIT_ALIGN_TOP = min(top-aligns)
        # Bottom_budget[i] = INIT_ALIGN - bottom[i]
        # Top_budget[i] = top[i] - INIT_ALIGN_TOP

        # Align on: self.init_align
        self.windows = {var: [b, u] for b, u, var in zip(bottom_aligns, up_locs, self.variables)}
        self.budgets = {var: [b, u] for b, u, var in zip([self.init_align - bot for bot in bottom_aligns],
                                                         [top - min_up for top in up_locs], self.variables)}
        self.initial_align = {var: bottom[0] for var, bottom in zip(self.windows.keys(), self.windows.values())}


class DatasetPlanner:
    def __init__(self, raw_code, vector_var, name):
        self.raw = raw_code
        self.specs = {"name": name,
                      "x name": "x",
                      "y name": "y",
                      "compression": False,
                      "error": None,
                      "split files": None}
        self.logs = LogManager()
        self.vec_vars: VariableVectorManager = vector_var

        self.aligner = None

    def __data_row_desc(self, vs, align: str) -> list[str]:
        """
        Generate a description for the variables given as input according
        to the windows kept by the aligner authority.
        # TODO: docs

        :param vs: requested variables
        :param align: generic index aligning token
        :return: a list containing three rows, as specified above
        """

        c_tok, x_tok = '<----> ', '<-^^-> '
        n_f = d_f = g_f = "> "
        for var in vs:
            l_wn, t_wn = self.aligner.window_generation_type[var]
            b_wn = c_tok * l_wn if l_wn < 4 else c_tok + f"..{l_wn - 2}.." + c_tok
            u_wn = c_tok * t_wn if t_wn < 4 else c_tok + f"..{t_wn - 2}.. " + c_tok
            r = b_wn + x_tok + u_wn
            v_l = f"| {var} "
            w_l = "| " + (f"{l_wn} before x" if b_wn else "").ljust(len(b_wn), ' ') + f"{align.upper()}   " + \
                  (f" {t_wn} before {align.lower()}" if t_wn else "").ljust(len(u_wn), ' ')
            max_l = max(len(r), len(v_l), len(w_l))
            r, w_l, v_l = r.ljust(max_l), w_l.ljust(max_l), v_l.ljust(max_l)
            n_f += v_l
            d_f += w_l
            g_f += r

        return [n_f, d_f, g_f]

    def _gen_dataset_description(self):
        """
        Draws a graphical description of the model onto the log file. The description
        contains both the labels of the variable used, the window considered in the model and
        their order. The graph generated should be of help when trying to build data to feed into
        the model after training has completed.

        The actual string generation work is done by __data_row_desc.

        :return: Logs a graphical description of the model.
        """

        # MULTI TARGET INCAPABLE!!
        t_vars = [self.aligner.target_var]
        nt_vars = [v for v in self.aligner.variables if v not in t_vars]

        x_desc, y_desc = self.__data_row_desc(nt_vars, 'x'), self.__data_row_desc(t_vars, 'y')
        # Log both target and non target variables description onto the log file
        self.logs.log_n(["\n> DATA FED INTO THE MODEL:\n"] + x_desc)
        self.logs.log_n(["\n> EXPECTED OUTPUT OF THE MODEL:\n"] + y_desc)

    def _parse_window_request_statement(self, statement):
        request = [0, 0]
        try:
            var_name = statement.split('from')[1].strip()
        except IndexError:
            raise BadWindowRequest(statement, "Missing data source ('from' keyword) ")
        # Composite statement of the type
        # "take <> before x and take <> after x
        if 'and' in statement:
            request[0] = int(statement.split('take')[1].split('before')[0].strip())
            request[1] = int(statement.split('and')[1].split('after')[0].strip())
        elif 'before' in statement:
            request[0] = int(statement.split('take')[1].split('before')[0].strip())
        elif 'after' in statement:
            request[1] = int(statement.split('take')[1].split('after')[0].strip())
        else:
            self.aligner.singleton(var_name)
            return

        self.aligner.wants(var_name, request)

    def parse(self):

        end_of_decl = self.raw.index('end plan\n')

        # Get preferred error mode from beginning of declaration, if any
        if 'expecting' in self.raw[0]:
            error_mode = self.raw[0].split('expecting')[1].strip()
            self.change_field("error", error_mode)

        if sum([statement.count('}') + statement.count('{') for statement in self.raw]) != 2:
            raise ParenthesisMismatch(self.specs['name'])

        align_vars = []
        align_factors = []
        align_format = alignment_mode = target_variable = None

        plan = [li for li in self.raw[1:end_of_decl] if li not in {'{\n', '}\n'} and not str.isspace(li)]

        for statement in plan:

            statement = statement.strip() if '#' not in statement else statement.split('#')[0].strip()

            if 'align' in statement:
                try:
                    align_vars += [v.strip() for v in statement.split('align')[1].split('against')[0].split(',')]
                    align_factors += [f.strip() for f in statement.split('against')[1].split('as')[0].split(',')]
                    alignment_mode = statement.split('as')[1].strip()

                    if 'format' in statement:
                        alignment_mode = alignment_mode.split('with')[0].strip()
                        align_format = statement.split('format')[1].strip()
                except Exception:
                    raise BadAlignmentCommand(statement)
            elif 'consider x' in statement:
                # A consider statement is unique per model declaration and should only
                # occur after every alignment has been completed. Note that
                # variables are permanently altered by alignment. That is to be expected
                # as they have gone beyond sap-scope
                if not alignment_mode or alignment_mode == 'date' and not align_format:
                    raise BadAlignmentCommand(statement, "No alignment mode provided or no align format provided for "
                                                         "date alignment")

                self.aligner = Aligner(self.vec_vars, align_vars, align_factors)
            elif 'make' in statement:
                target_variable = statement.split('make')[1].split('the')[0].strip()

                # Immediate vectorial model output request recognized
                if 'take' in statement:
                    self._parse_window_request_statement(statement.split('and')[1].strip())

            elif 'pair' in statement:
                if not self.aligner:
                    raise BadAlignmentCommand(statement, "Plan and x declaration must precede the binding of x and "
                                                         "the target")

                self.aligner.add_target(target_variable)

            else:
                # Else assume statement is a composite description
                # of a planning procedure. Just throw an error if nothing matches
                # a known expression. Again, error mode does not apply to semantic.
                if 'take' not in statement:
                    raise BadAlignmentCommand(statement, f"Incorrect planning statement: \"{statement}\"")

                self._parse_window_request_statement(statement)

    def change_field(self, field_name, new_val):
        """
        Change the requested field inside the model. Any label can be accepted
        as a field, but a limited number of those are actually recognized.
        :param field_name: the name of the field to be changed
        :param new_val: the new value of the field

        Usage:

        >> set ModelName compression = gzip
        """
        if field_name and new_val:
            self.logs.log(f"> Changing field {field_name} to {new_val}")
            self.specs[field_name] = new_val
        else:
            # Error mode does not apply for semantic errors.
            raise IncorrectFieldName(field_name)

    def _save_model_data(self, x_data, y_data, path):
        """
        Saves the model data as specified. Note that if no file is
        specified by the
        >> compile <Model> into <File>
        directive, an attempt to partition the data according to a precedent
        >> split <Model> into [...]
        is made. If that fails, if the model has requested to attempt recovery, a fallback
        file is searched.
        Note that specifying an output file in the compile directive OVERWRITES any
        attempt to partition the dataset.

        On a technical note, partitions aren't actually assured to be of the exact
        percentage, as numpy indexing requires a cast to int following the call to np.ceil.
        However, it is ensured that the whole data vector is partitioned, as
        it is numerically sound that ceil(100/100.0 * len(x_data[0])+1) == len(x_data[0])+1.

        :param x_data: the input data generated by self.compile()
        :param y_data: the desired output data generated by self.compile()
        :param path: a path to the save file or None
        """
        try:
            if path:
                # np.save_z() takes the set name as a key in its call, make a dictionary and
                # use double star operator **dic to assign user-indicated names
                try:
                    numpy.savez(path, **{self.specs['x name']: x_data, self.specs['y name']: y_data})
                    self.logs.log("> Successfully compiled the plan as a (x,y) pair with labels " +
                                  self.specs['x name'] + " e " + self.specs['y name'])

                except FileNotFoundError or FileExistsError:
                    raise DatasetFileError(self.specs['name'])

            elif self.specs['split files'] is not None and self.specs['proportions']:
                """ 
                Make partitions by getting the percentages, which are assured to
                sum up to 1, normalizing them and getting their cumulative sum. The 
                actual job is done by np.split, which gets a list of indexes and 
                cuts the array along those splitting points
                Example:
                >> a = [1, 2, 3, 4, 5]
                >> np.split(a, [1, 2])
                >> [1], [2], [3, 4, 5]
                """
                props: list = self.specs['proportions']

                sections = np.ceil(np.cumsum(np.array(props[:-1]) / 100.0) * np.size(x_data, axis=0) + 1).astype(
                    np.int32)
                x_s, y_s = np.split(x_data, sections), np.split(y_data, sections)
                names: list = [{self.specs['x name']: x, self.specs['y name']: y} for x, y in zip(x_s, y_s)]

                # (At this point, self.spect cannot be None)
                # noinspection PyTypeChecker
                [np.savez(file, **n) for file, n in zip(self.specs['split files'], names)]

                self.logs.log("> Successfully compiled the plan as a list of (x,y) pairs with labels \"" +
                              self.specs['x name'] + "\" e \"" + self.specs['y name'] + "\" split with " +
                              " percentages " + str(self.specs['proportions']))

            else:
                raise DatasetFileError("No path was specified nor a partition was assigned to the model.")

        except Exception as BroadException:
            if self.specs['error'] == 'attempt_recovery' and self.specs['fallback file']:
                self.logs.log(f"> Exception occurred: {BroadException.__str__()}.\n"
                              f"The attempt to partition the model into save files has failed. The entirety "
                              "of the data has been dumped with default fallback labels 'x' and 'y' in the "
                              "fallback file.")
                # Save data into fallback file
                np.savez(self.specs['fallback file'], **{'x': x_data, 'y': y_data})
            else:
                raise BadSplitRequest("NA", "A fatal error occurred, failed to split result into"
                                            " requested partitions.")

    def compile(self, save_file):

        """
        Compiles a plan into a dataset, applying the specified transformations to it
        and storing it according to instructions given.
        The actual numeric work is done by the aligner children, while the planner
        just takes care of the house-keeping functions (storing, compressing...)
        :param save_file the label of a model file defined in precedence.
        :return: A dataset in the requested location.

        Example
        >> compile <Model name> into <Model File>
        """

        m_lb, m_ub = max(self.aligner.lower_bound), min(self.aligner.upper_bound)

        # The number of data points generated are given by m_ub - m_lb. For more information
        # see the documentation is the Aligner object
        n_data_points = m_ub - m_lb + 1
        self.logs.log(f"> Generated {n_data_points} data points from input configuration")

        no_target_v = [v for v in self.aligner.variables if v != self.aligner.target_var]
        self.logs.log(f"> Non target variables:" + str(no_target_v) +
                      f", Target variables: {self.aligner.target_var}")
        try:
            x_data = np.hstack([self.aligner.get_convolution(var) for var in no_target_v])
            y_data = self.aligner.get_convolution(self.aligner.target_var)
        except ValueError:
            raise DatasetInternalError(self.specs['name'])

        self._save_model_data(x_data, y_data, save_file)
        self._gen_dataset_description()

        return

    def log(self, log_file_path):
        self.logs.write_logs(log_file_path)
