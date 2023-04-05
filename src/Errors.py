from datetime import datetime

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


class BadFormatStyle(Exception):
    def __init__(self, path, error_specific):
        self.ErrorPath = path
        self.ErrorSpecific = error_specific

    def __str__(self) -> str:
        self.TimeOccurred = datetime.now()
        return "IncorrectFormatFile Error from Input: <" + \
            str(self.ErrorPath) + \
            " > at " + str(self.TimeOccurred) + "\n" + str(self.ErrorSpecific)
