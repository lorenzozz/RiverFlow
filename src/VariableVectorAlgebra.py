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
        try:
            self.variables[var_name] = np.array(py_data).astype(as_type)
        # Solve missing datapoint by just putting them equal to zero.
        except ValueError:
            intermediate = np.array(py_data)
            intermediate[intermediate == ''] = '0'

            self.variables[var_name] = intermediate.astype(as_type)

        self.variables_dims[var_name] = np.size(self.variables[var_name])
        if self.grammar:
            self.grammar[var_name] = self.variables[var_name]

    def add_copy_of(self, variable_to_copy, new_label):
        # Performs a deep copy of variable <variable_to_copy> and adds
        # a new variable under label new_label
        self.variables[variable_to_copy] = np.ndarray.copy(self.variables[new_label])
        self.variables_dims[variable_to_copy] = np.size(self.variables[new_label])
        if self.grammar:
            self.grammar[variable_to_copy] = self.variables[variable_to_copy]

    def execute_line(self, statement, line_number):
        # First parse statement into lh-side, rh-side
        if '=' not in statement and 'print(' not in statement:
            self.error_issuer(NotAssignmentExpression, f"At line {line_number}: Attempting" +
                              " to execute a non-assignment expression is an error as" +
                              "every variable is treated by value, not reference")
        try:
            if 'print(' in statement:
                print(eval(statement.split('print(')[1].split(')')[0].strip(), self.grammar))
            else:
                ref_var = statement.split('=')[0].strip()
                action = statement.split('=')[1].strip()

                self.variables[ref_var] = eval(action, self.grammar)
                self.variables_dims[ref_var] = np.size(self.variables[ref_var])
                self.grammar[ref_var] = self.variables[ref_var]

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

        # Predefined function aliases.
        self.grammar = \
            {"discretizza": vec_discrete,
             "aggiungi_rumore": vec_add_noise,
             "da_categorico_a_numero": vec_bool_to_num,
             "media_zero": vec_zero_mean,
             "media": vec_mean,
             "dev_stand": vec_std,
             "azzera_outlier": vec_zero_outliers,
             "interpola_outlier": vec_inter_outlier,
             "interval": vec_interval,
             "lunghezza": np.size,
             "gaussiana": "gaussiana",  # Label
             "esponenziale": "esponenziale",  # Label
             "uniforme": "uniforme",   # Label
             }
        self.grammar.update(self.variables)

    @staticmethod
    def take_type(var_type):
        return {"categorical": str,
                "boolean": str,
                "numeric": numpy.float64}[var_type]

    def add_package(self, package, pack):
        # Add passed package to grammar
        self.grammar[package] = pack

    def get_sizeof_var(self, variable):
        return self.variables_dims[variable]
