class ValidateCallWrapperWrapper:
    def __init__(self, wrapped):
        self.orig = wrapped

    def __eq__(self, other):
        return self.raw_function == other.raw_function

    @property
    def raw_function(self):
        return getattr(self.orig, 'raw_function') or self.orig


def same_method(m1, m2) -> bool:
    """
    Return whether m1 and m2 are the same method.

    Workaround for pydantic/pydantic#6390.
    """
    return ValidateCallWrapperWrapper(m1) == ValidateCallWrapperWrapper(m2)
