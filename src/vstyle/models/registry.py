"""Name -> factory registries, so a new model is registered instead of added to an if/elif chain."""
from collections.abc import Callable


class Registry:
    def __init__(self, kind: str):
        self.kind, self._items = kind, {}

    def register(self, name: str) -> Callable:
        def decorator(factory):
            self._items[name] = factory
            return factory
        return decorator

    def create(self, name: str, *args, **kwargs):
        if name not in self._items:
            raise ValueError(f"unknown {self.kind} {name}")
        return self._items[name](*args, **kwargs)

    def names(self) -> tuple:
        return tuple(self._items)

    def __contains__(self, name):
        return name in self._items


ESTIMATORS = Registry("tabular model")
