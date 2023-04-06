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
