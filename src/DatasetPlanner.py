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
        for bound in range(2):
            if request[bound] > self.budgets[var_name][bound]:
                self.windows[var_name][bound] -= request[bound] - budget[bound]
                self.budgets[var_name][bound] = 0
            else:
                self.budgets[var_name][bound] -= request[bound]

    def show_status(self):
        print(self.windows)
        print(self.budgets)

    def add_target(self, target_var):
        self.target_var = target_var

    def singleton(self, var_name):
        if var_name in self.window_generation_type.keys():
            raise VariableSliceRedefinition(var_name)
        self.window_generation_type[var_name] = [0, 0]

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
        up_locs = [size + bottom for size, bottom in zip(sizes, bottom_aligns)]
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

        self.windows = {var: [b, u] for b, u, var in zip(bottom_aligns, up_locs, self.variables)}
        self.budgets = {var: [b, u] for b, u, var in zip([self.init_align - bot for bot in bottom_aligns],
                                                         [top - min_up for top in up_locs], self.variables)}


class DatasetPlanner:
    def __init__(self, raw_code, vector_var, name):
        self.raw = raw_code
        self.specs = {"x name": None,
                      "y name": None,
                      "compression": False,
                      "error": None}
        self.logs = LogManager()
        self.vec_vars: VariableVectorManager = vector_var
        self.model_name = name

        self.aligner = None

    def parse(self):

        end_of_decl = self.raw.index('end plan\n')

        # Get preferred error mode from beginning of declaration, if any
        if 'expecting' in self.raw[0]:
            error_mode = self.raw[0].split('expecting')[1].strip()
            self.change_field("error", error_mode)

        if sum([statement.count('}') + statement.count('{') for statement in self.raw]) != 2:
            raise ParenthesisMismatch(self.model_name)

        align_vars = []
        align_factors = []
        align_format = None
        alignment_mode = None
        target_variable = None

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

                request = [0, 0]
                var_name = statement.split('from')[1].strip()
                if 'before' in statement:
                    request[0] = int(statement.split('take')[1].split('before')[0].strip())
                if 'after' in statement:
                    request[1] = int(statement.split('and')[1].split('after')[0].strip())
                else:
                    self.aligner.singleton(var_name)
                    continue

                self.aligner.wants(var_name, request)

    def change_field(self, field_name, new_val):

        if field_name in self.specs:
            self.specs[field_name] = new_val
        else:
            # Error mode does not apply for semantic errors.
            raise IncorrectFieldName(field_name)

    def compile(self):

        

    def log(self, log_file_path):
        self.logs.write_logs(log_file_path)
