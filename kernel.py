from typing import Any
from dataclasses import dataclass

@dataclass
class Kernel:
    bin: Any

class KernelCache:
    def __init__(self):
        self.kerns: dict[str, Kernel] = dict()

    def save(self, name, kern):
        self.kerns[name] = kern

    def has(self, name):
        return name in self.kerns

    def get(self, name):
        return self.kerns[name]

