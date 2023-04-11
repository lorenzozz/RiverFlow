import warnings


def issue_error(error_obj, additional):
    raise (error_obj(additional))


def issue_warning(error_obj, additional):
    warnings.warn(additional, error_obj)


class NotAssignmentExpression(Exception):
    def __init__(self, expression):
        self.expression = expression

    def __str__(self):
        return self.expression


class UnrecognizedPackageReference(Exception):
    def __init__(self, expression):
        self.expression = expression

    def __str__(self):
        return self.expression


class IncorrectPlotRequest(Exception):
    def __init__(self, x, y, additional):
        self.x = x
        self.y = y
        self.additional = additional

    def __str__(self):
        return f"Cannot plot {self.x} against {self.y}: " + self.additional


class VariableTypeUnspecified(Exception):
    def __init__(self, expression):
        self.expression = expression

    def __str__(self):
        return self.expression


class IncompatibleTargetSize(Exception):
    def __init__(self, expression):
        self.expression = expression

    def __str__(self):
        return self.expression
