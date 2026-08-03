from typing import Any
from dataclasses import dataclass

@dataclass
class Kernel:
    bin: Any

class KernelCache:
    def __init__(self):
        self.kerns: dict[str, Kernel] = dict()

    def save(self, name: str, kern: Kernel):
        self.kerns[name] = kern

    def has(self, name: str):
        return name in self.kerns

    def get(self, name: str):
        return self.kerns[name]

