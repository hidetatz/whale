import ctypes

class DType:
    def __repr__(self):
        return self.__class__.__name__.lower()
    def is_float(self): return isinstance(self, Float32) or isinstance(self, Float64)
    def is_int(self): return isinstance(self, Int32) or isinstance(self, Int64)

class Int32(DType):
    def ctype(self): return ctypes.c_int32

class Int64(DType):
    def ctype(self): return ctypes.c_int64

class Float32(DType):
    def ctype(self): return ctypes.c_float

class Float64(DType):
    def ctype(self): return ctypes.c_double

int32 = Int32()
int64 = Int64()
float32 = Float32()
float64 = Float64()
