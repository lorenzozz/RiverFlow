from datetime import datetime   # datetime.now


class IncorrectLogFile(Exception):

    def __init__(self, path):
        self.ErrorPath = path

    def __str__(self) -> str:
        self.TimeOccurred = datetime.now()
        return "IncorrectLogFile Error from Input: <" + \
            str(self.ErrorPath) + \
            " > at " + str(self.TimeOccurred)


class IncorrectDataFile(Exception):
    def __init__(self, path):
        self.ErrorPath = path

    def __str__(self) -> str:
        self.TimeOccurred = datetime.now.time()
        return "IncorrectDataFile Error from Input: <" + \
            str(self.ErrorPath) + \
            " > at " + str(self.TimeOccurred)


class IncorrectFormatFile(Exception):
    def __init__(self, path):
        self.ErrorPath = path

    def __str__(self) -> str:
        self.TimeOccurred = datetime.now.time()
        return "IncorrectFormatFile Error from Input: <" + \
            str(self.ErrorPath) + \
            " > at " + str(self.TimeOccurred)


# Generic bad format style sheet error
class BadFormatStyle(Exception):
    def __init__(self, path, error_specific):
        self.ErrorPath = path
        self.ErrorSpecific = error_specific

    def __str__(self) -> str:
        self.TimeOccurred = datetime.now()
        return "IncorrectFormatFile Error from Input: <" + \
            str(self.ErrorPath) + \
            " > at " + str(self.TimeOccurred) + "\n" + str(self.ErrorSpecific)


# Descriptive alias for specific mismatched parenthesis error
class MismatchedToken(BadFormatStyle):
    def __init__(self, path, error_specific):
        super().__init__(path, error_specific)

    def __str__(self):
        return super().__str__()


# Descriptive alias for specific missing section error
class MissingSection(BadFormatStyle):
    def __init__(self, path, error_specific):
        super().__init__(path, error_specific)

    def __str__(self):
        return super().__str__()
