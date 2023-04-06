import numpy

from MathErrors import *
from VariableMath import *

import numpy as np


class VariableVectorManager:

    def __init__(self):
        # Dictionary containing all <var_name>:<numpy_array> pairs
        self.variables: dict = {}
        # Dictionary containing all <var_name>:<vec_dimension> to
        # match when doing vectorial operations.
        self.variables_dims: dict = {}

        self.error_issuer = issue_error
        self.grammar = None

    def add_variable(self, var_name, py_data, as_type):
        self.variables[var_name] = np.array(py_data).astype(as_type)
        self.variables_dims[var_name] = np.size(self.variables[var_name])
        if self.grammar:
            self.grammar[var_name] = self.variables[var_name]

    def add_copy_of(self, variable_to_copy, new_label):
        # Performs a deep copy of variable <variable_to_copy> and adds
        # a new variable under label new_label
        self.variables[variable_to_copy] = np.ndarray.copy(self.variables[new_label])
        self.variables_dims[variable_to_copy] = np.size(self.variables[new_label])

    def execute_line(self, statement, line_number):
        # First parse statement into lh-side, rh-side
        if '=' not in statement:
            self.error_issuer(NotAssignmentExpression, f"At line {line_number}: Attempting" +
                              " to execute a non-assignment expression is an error as" +
                              "every variable is treated by value, not reference")
        try:
            ref_var = statement.split('=')[0].strip()
            action = statement.split('=')[1].strip()

            print("> Executing statement: ", statement)
            print(f"> Parsed as {ref_var}, {action}")
            self.variables[ref_var] = eval(action, self.grammar)

        except Exception as BroadException:
            NotImplemented
            print(BroadException)

    def get_variable(self, variable_name):
        return self.variables[variable_name]

    @property
    def variable_amt(self):
        return len(self.variables.keys())

    def change_error_mode(self, no_except):
        if no_except:
            self.error_issuer = issue_warning
        else:
            self.error_issuer = issue_error

    def load_grammar_mapper(self):

        self.grammar = \
            {"Discretizza": vec_discrete,
             "AggiungiRumore": vec_add_noise,
             "BoolANumero": vec_bool_to_num,
             "MediaZero": vec_zero_mean,
             "Media": vec_mean,
             "DevStand": vec_std,
             "AzzeraOutlier": vec_zero_outliers,
             "InterpolaOutlier": vec_inter_outlier
             }
        self.grammar.update(self.variables)

    def take_type(self, var_type):
        return {"Categorical": str,
                 "Boolean": str,
                 "Numeric": numpy.float64}[var_type]