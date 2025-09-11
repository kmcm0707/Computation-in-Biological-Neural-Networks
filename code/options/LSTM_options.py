from options.complex_options import operatorEnum


class lstmOptions:
    """
    Options for the complex synapse and individual complex synapse
    """

    def __init__(
        self,
        update_rules=None,
        operator: operatorEnum = operatorEnum.mode_6,
    ):
        self.update_rules = update_rules
        self.operator = operator

    def __str__(self):
        string = ""
        for key, value in vars(self).items():
            string += f"{key}: {value}\n"
        return string
