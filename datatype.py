import ctypes

class dtype:
    def __repr__(self):
        return self.__class__.__name__.lower()
    def is_float(self): return isinstance(self, (Float32, Float64))
    def is_int(self): return isinstance(self, (Int32, Int64))

class Int32(dtype):
    def ctype(self): return ctypes.c_int32

class Int64(dtype):
    def ctype(self): return ctypes.c_int64

class Float32(dtype):
    def ctype(self): return ctypes.c_float

class Float64(dtype):
    def ctype(self): return ctypes.c_double

int32 = Int32()
int64 = Int64()
float32 = Float32()
float64 = Float64()
