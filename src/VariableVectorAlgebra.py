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

    def add_variable(self, var_name, py_data):
        self.variables[var_name] = np.array(py_data)
        self.variables_dims[var_name] = np.size(self.variables[var_name])

    def add_copy_of(self, variable_to_copy, new_label):
        # Performs a deep copy of variable <variable_to_copy> and adds
        # a new variable under label new_label
        self.variables[variable_to_copy] = np.ndarray.copy(self.variables[new_label])
        self.variables_dims[variable_to_copy] = np.size(self.variables[new_label])

    def execute_function(self, statement, no_except):
        print(self.variables["Month"])
        try:
            eval(statement, self.grammar)
        except Exception as BroadException:
            NotImplemented
            print(BroadException)
        print(statement, no_except)

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
             "Normalizza": vec_normalize,
             "Media": vec_mean,
             "DevStand": vec_std,
             "AzzeraOutlier": vec_zero_outliers,
             "InterpolaOutlier": vec_inter_outlier
             }
        self.grammar.update(self.variables)

        print(self.grammar)