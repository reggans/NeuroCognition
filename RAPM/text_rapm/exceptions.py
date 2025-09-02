class IncompatibleAttributesError(Exception):
    def __init__(self, reasons):
        super().__init__("Incompatible axis attributes")
        self.reasons = reasons


class GenerationFailure(Exception):
    pass


class DistractorFailure(Exception):
    pass
