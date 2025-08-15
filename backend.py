import os
from enum import Enum, auto


class Backend(Enum):
    PYTHON = auto()
    CUDA = auto()

    @classmethod
    def detect(cls):
        be = os.environ.get("WHALE_BACKEND")
        if be is None:

            def cmd_avail(cmd):
                return os.system(f"command -v {cmd} > /dev/null") == 0

            # auto detect
            if cmd_avail("nvcc"):
                return cls.CUDA

            return cls.PYTHON

        if be == "CUDA":
            return cls.CUDA

        return cls.PYTHON
