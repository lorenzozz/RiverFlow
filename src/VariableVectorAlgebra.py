from MathErrors import *
from VariableMath import *
from difflib import get_close_matches  # Neat error descriptions


class VariableVectorManager:

    def __init__(self):
        # Dictionary containing all <var_name>:<numpy_array> pairs
        self.variables: dict = {}
        # Dictionary containing all <var_name>:<vec_dimension> to
        # match when doing vectorial operations.
        self.variables_dims: dict = {}

        self.error_issuer = issue_error
        self.env = None

    def add_variable(self, var_name, py_data, as_type):
        try:
            self.variables[var_name] = np.array(py_data).astype(as_type)
        # Solve missing datapoint by just putting them equal to zero.
        except ValueError:
            intermediate = np.array(py_data)
            # Naive attempt to counter missing data, should be more modular
            # in this sense.
            intermediate[intermediate == ''] = '0'

            self.variables[var_name] = intermediate.astype(as_type)

        self.variables_dims[var_name] = np.size(self.variables[var_name])
        if self.env is not None:
            self.env[var_name] = self.variables[var_name]

    def add_copy_of(self, variable_to_copy, new_label):
        # Performs a deep copy of variable <variable_to_copy> and adds
        # a new variable under label new_label
        self.variables[variable_to_copy] = np.ndarray.copy(self.variables[new_label])
        self.variables_dims[variable_to_copy] = np.size(self.variables[new_label])
        if self.env:
            self.env[variable_to_copy] = self.variables[variable_to_copy]

    def execute_line(self, statement, line_number):
        # First parse statement into lh-side, rh-side
        if '=' not in statement and 'print(' not in statement:
            self.error_issuer(NotAssignmentExpression, f"Missing '=' at line {line_number}: " +
                              f"Attempting to execute a non-assignment expression is an " +
                              f"error as every variable is taken by value, not reference")
        try:
            if 'print(' in statement:
                print("* ", eval(statement.split('print(')[1].split(')')[0].strip(), self.env))
            else:
                ref_var = statement.split('=', 1)[0].strip()
                action = statement.split('=', 1)[1].strip()

                # Enforce 'new' syntax.
                if ref_var not in self.env.keys():
                    raise NameError(ref_var)

                self.variables[ref_var] = eval(action, self.env)
                self.variables_dims[ref_var] = np.size(self.variables[ref_var])
                self.env[ref_var] = self.variables[ref_var]

        except Exception as BroadException:
            # Attempt to give user a reasonable description of the occurred error
            err_expl = self.get_useful_error_description(BroadException, statement)
            raise GenericMathError(err_expl)

    def get_useful_error_description(self, exception, statement):
        """
        Provides the user with a useful error description from the exception
        that occurred inside the .act section, as specified from the error mode
        (last part not implemented yet)

        :param exception: exception to clarify
        :param statement: statement where exception occurred
        :return: a useful description of the error or the original exception
         __str()__ if the program cannot find a feasible explanation for the error
        """

        # Unrecognized name referenced
        if isinstance(exception, NameError):
            possible_matches = get_close_matches(exception.__str__(), list(self.env.keys()))
            error_expl = f"Name <{exception.__str__()}> in statement \"{statement.strip()}\" not recognized. " + \
                         (f"Did you mean one of the following? {possible_matches}" if possible_matches
                          else "")

        else:
            return exception.__str__()

        return error_expl

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
        """
        Defines the environment in which the .make section runs inside
        the python eval() virtual machine. Other than the environment defined
        below, it is possible to extend it making use of the
            import <recognized_packet> as <alias>
            or
            import <recognized_packet>
        statement.
        """

        self.env = {
            # Primitive functions
            "discretizza": vec_discrete,
            "aggiungi_rumore": vec_add_noise,
            "da_categorico_a_numero": vec_bool_to_num,
            "shuffle": vec_shuffle,
            "media_zero": vec_zero_mean,
            "media": vec_mean,
            "zero_con_probabilita": vec_zero_with_prob,
            "dev_stand": vec_std,
            "azzera_outlier": vec_zero_outliers,
            "interpola_outlier": vec_inter_outlier,
            "intervallo": vec_interval,
            "lunghezza": np.size,
            "tronca": vec_truncate,
            "one_hot_encode": vec_one_hot,
            "stack": vec_stack,
            "load_vec": vec_load,
            "z_score":  vec_zscore,
            "max_linear": vec_max_linear,
            "zavadskas": vec_zavadskas,
            "profile": vec_profile,

            # Local functional data

            "gaussiana": "gaussiana",  # Gaussian np.normal distribution
            "esponenziale": "esponenziale",  # Exponential np.exponential distribution
            "uniforme": "uniforme",  # Uniform np.uniform distribution

            # Keyword definitions
        }
        # Enhance environment with user-defined variables.
        self.env.update(self.variables)

    @staticmethod
    def take_type(var_type: str, variable: str = ""):
        """ Get python/numpy equivalent type of makefile variable type.

        :param var_type: The type requested
        :param variable: The variable name (Optional to generate more explicative errors)
        :return: The python equivalent of requested variable type
        """

        recognized_v_types = {
            "categorical": str,
            "boolean": str,
            "numeric": np.float64,
            "integer": np.int32
        }
        if var_type not in recognized_v_types.keys():
            raise TypeError("Expected user defined variable {v} to have a "
                            "predefined type, found unrecognized type {t}"
                            "instead.".format(v=variable, t=var_type))
        else:
            return recognized_v_types[var_type]

    def add_package(self, package, pack):
        # Add passed package to grammar
        self.env[package] = pack

    def get_sizeof_var(self, variable):
        return self.variables_dims[variable]
