from ctypes import c_int, c_longlong, c_float, c_double

class DType:
    def __repr__(self):
        return self.__class__.__name__.lower()

class Int32(DType):
    def ctype(self): return c_int

class Int64(DType):
    def ctype(self): return c_longlong

class Float32(DType):
    def ctype(self): return c_float

class Float64(DType):
    def ctype(self): return c_double

int32 = Int32()
int64 = Int64()
float32 = Float32()
float64 = Float64()
