from DatasetErrors import*
from LogManager import*


class DatasetPlanner:
    def __init__(self, raw_code, vector_var):
        self.raw = raw_code
        self.specs = {"x name": None,
                      "y name": None,
                      "compression": False,
                      "error": None}
        self.logs = LogManager()
        self.vec_vars = vector_var

    def parse(self):
        NotImplemented


    def change_field(self, field_name, new_val):

        if field_name in self.specs:
            self.specs[field_name] = new_val
        else:
            # Error mode does not apply for semantic errors.
            raise IncorrectFieldName(field_name)

    def compile(self, compression = False):
        NotImplemented

    def log(self, log_file_path):
        self.logs.write_logs(log_file_path)


