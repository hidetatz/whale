from abc import ABC, abstractmethod

class Compiler(ABC):
    @abstractmethod
    def compile(self, name: str, code: str): ...
