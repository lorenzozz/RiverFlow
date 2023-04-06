class IncorrectFieldName(Exception):
    def __init__(self, field_name):
        self.field_name = field_name

    def __str__(self):
        return f"Unknown dataset field referenced: {self.field_name}"


class BadAlignmentCommand(Exception):
    def __init__(self, command, specific=""):
        self.command = command
        self.specific = specific

    def __str__(self):
        return f"Incorrect align command: {self.command}" + self.specific


class ParenthesisMismatch(Exception):
    def __init__(self, model_name):
        self.model_name = model_name

    def __str__(self):
        return "Bad expression delimiters '}' in model <" + str(self.model_name) + "> declaration"


class VariableSliceRedefinition(Exception):
    def __init__(self, var_id):
        self.var_id = var_id

    def __str__(self):
        return "Multiple slice values for the same variable: " + selfvar_id