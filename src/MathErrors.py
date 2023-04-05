import warnings


def issue_error(error_obj, additional):
    raise (error_obj(additional))


def issue_warning(error_obj, additional):
    warnings.warn(additional, error_obj)
