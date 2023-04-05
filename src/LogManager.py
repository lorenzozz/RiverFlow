from Errors import IncorrectLogFile

LogFile = 'C:/Users/picul/OneDrive/Documenti/RiverLogFile.txt'
TextSeparator = '**********************************************\n'


class LogManager:
    def __init__(self, log_file):
        try:
            self.log_file = log_file
            self.log_element = open(LogFile, 'w')
        except Exception as BroadError:
            raise IncorrectLogFile(self.log_element)

    def close(self):
        self.log_element.close()

    def log(self, text):
        try:
            self.log_element.write(text + '\n')
        except Exception as GenericError:
            self.close()
            raise IncorrectLogFile(self.log_file)

    def separate(self):
        self.log(TextSeparator)

    def log_n(self, *var_arg):
        for arg in var_arg:
            self.log(arg)
