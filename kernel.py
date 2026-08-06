from typing import Any
from dataclasses import dataclass

@dataclass
class Kernel:
    bin: Any

class KernelCache:
    def __init__(self):
        self.kerns: dict[str, Kernel] = dict()
        self.hitcnt: dict[str, int] = dict()

    def save(self, name: str, kern: Kernel):
        self.kerns[name] = kern
        self.hitcnt[name] = 0

    def has(self, name: str):
        return name in self.kerns

    def hit(self, name: str):
        self.hitcnt[name] += 1

    def get(self, name: str):
        return self.kerns[name]

