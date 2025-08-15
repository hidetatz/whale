import ctypes
from dataclasses import dataclass

@dataclass
class CPUMemoryBuffer:
    raw: list[float] # python buffer

@dataclass
class DeviceMemoryBuffer:
    ptr: ctypes.c_void_p  # pointer on device memory
    length: int  # array length
    size: int  # byte size

class Device:
    def allocate(self, length: int): raise NotImplementedError()
    def free(self, dev_buff: DeviceMemoryBuffer): raise NotImplementedError()
    def copy_to_device(self, cpu_buff: CPUMemoryBuffer, dev_buff: DeviceMemoryBuffer): raise NotImplementedError()
    def copy_from_device(self, dev_buff: DeviceMemoryBuffer): raise NotImplementedError()

