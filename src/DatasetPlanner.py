from DatasetErrors import *
from LogManager import *
import numpy as np  # Vector support, sliding view.
import time  # dates comparison


class Aligner:

    def __init__(self, variables, aligning_variables):
        self.variables = variables
        self.alignment = aligning_variables

        self.windows = [(0, 0) for _ in variables]
        # Upper alignment budget and Lower alignment budget are stored in the same tuple
        self.budgets = [(0, 0) for _ in variables]

    @staticmethod
    def date_equal(*dates, string):
        time_dates = [time.strptime(date, string) for date in [*dates]]
        return all(time_dates[0] == other_date for other_date in time_dates)

    @staticmethod
    def index_equal(*numbers):
        all([numbers][0] == n_2 for n_2 in [numbers])

class DatasetPlanner:
    def __init__(self, raw_code, vector_var, name):
        self.raw = raw_code
        self.specs = {"x name": None,
                      "y name": None,
                      "compression": False,
                      "error": None}
        self.logs = LogManager()
        self.vec_vars = vector_var
        self.model_name = name

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

        for statement in self.raw[:end_of_decl]:
            print(statement)
            if 'align' in statement:
                try:
                    align_vars.append([v.strip() for v in statement.split('align')[1].split('against')[0].split(',')])
                    align_factors.append([f.strip() for f in statement.split('against')[1].split('as')[0].split(',')])
                    alignment_mode = statement.split('as')[1].strip()
                    if 'format':
                        alignment_mode = alignment_mode.split('with')[0].strip()
                        align_format = statement.split('format')[1].strip()
                except Exception:
                    raise BadAlignmentCommand(statement)

            if 'consider' in statement:
                # A consider statement is unique per model declaration and should only
                # occur after every alignment has been completed. Note that
                # variables are permanently altered by alignment. That is to be expected
                # as they have gone beyond sap-scope
                if not alignment_mode or alignment_mode == 'date' and not align_format:
                    raise BadAlignmentCommand(statement, "No alignment mode provided or no align format provided for "
                                                         "date alignment")


            else:
                # Else assume statement is a composite description
                # of a planning procedure. Just throw an error if nothing matches
                # a known expression. Again, error mode does not apply to semantic.
                NotImplemented

    def change_field(self, field_name, new_val):

        if field_name in self.specs:
            self.specs[field_name] = new_val
        else:
            # Error mode does not apply for semantic errors.
            raise IncorrectFieldName(field_name)

    def compile(self, compression=False):
        NotImplemented

    def log(self, log_file_path):
        self.logs.write_logs(log_file_path)
