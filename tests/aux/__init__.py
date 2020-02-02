def assert_that(obj, matcher):
    assert matcher(obj)


class ContractMatcher:
    def __init__(self, interface, properties):
        self._interface = interface
        self._properties = properties
        self._tested_obj = None
        self._missing_functions = []
        self._missing_properties = []

    def __call__(self, obj):
        self._tested_obj = obj
        self._collect_missing_functions()
        self._collect_missing_properties()
        return len(self._missing_functions) == 0 and len(self._missing_properties) == 0

    def _collect_missing_properties(self):
        for p in self._properties:
            if getattr(self._tested_obj, p) is None:
                self._missing_properties.append(p)

    def _collect_missing_functions(self):
        for func, args in self._interface:
            try:
                getattr(self._tested_obj, func)(*args)
            except (AttributeError, NotImplementedError):
                self._missing_functions.append(func)

    def __repr__(self):
        return f"the object {self._tested_obj}, is missing{self._print_missing()}"

    def _print_missing(self):
        message = ""
        if len(self._missing_functions):
            message += f" these functions: {', '.join(self._missing_functions)}"
        if len(self._missing_properties):
            if len(message) != 0:
                message += " and"
            message += f" these properties: {', '.join(self._missing_properties)}"
        return message


def follows_contract(interface=None, properties=None):
    return ContractMatcher(interface or [], properties or [])