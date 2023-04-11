from Errors import IncorrectLogFile


class LogManager:
    def __init__(self):
        self.log_buffer = ""

    def flush(self):
        self.log_buffer = ""

    def log(self, text):
        self.log_buffer += text + '\n'

    def separate(self):
        self.log("\n*************\n")

    def log_n(self, var_arg):
        for arg in var_arg:
            self.log(arg)

    def write_logs(self, into):
        try:
            with open(into, "w") as log_file:
                log_file.write(self.log_buffer)
        except FileNotFoundError:
            raise IncorrectLogFile(into)
