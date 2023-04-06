
class IncorrectFieldName(Exception):
    def __init__(self, field_name):
        self.field_name = field_name

    def __str__(self):
        return f"Unknown dataset field referenced: {self.field_name}"
