from __future__ import annotations

from ctypes import c_void_p
from dataclasses import dataclass
from enum import IntEnum, auto


def to_kern_name(code: OpCode, dim: int) -> str:
    return f"{code.__str__().lower()}_dim{dim}"


class OpCode(IntEnum):
    RECIP = auto()
    ADD = auto()
    MUL = auto()
    POW = auto()

    def __str__(self):
        return self.name


@dataclass
class Kernel:
    name: str
    src: str
    func_pointer: c_void_p  # on C side


class CodeGenerator:
    def generate_unary_kernel(self, code: OpCode, ndim: int):
        stride_params = [self.kern_param_ident(f"src_0_stride{i}", typ=int, const=True, memory="host") for i in range(ndim)]
        stride_params += [self.kern_param_ident(f"dst_stride{i}", typ=int, const=True, memory="host") for i in range(ndim)]
        params = (
            self.kern_param_ident("src_0_offset", typ=int, pointer=False, const=True, memory="host"),
            self.kern_param_ident("dst_offset", typ=int, pointer=False, const=True, memory="host"),
            *stride_params,
            self.kern_param_ident("src_0", typ=float, pointer=True, const=True, memory="device"),
            self.kern_param_ident("dst", typ=float, pointer=True, const=True, memory="device"),
        )
        return self.generate_kernel(code, ndim, 1, params)

    def generate_binary_kernel(self, code: OpCode, ndim: int):
        stride_params = [self.kern_param_ident(f"src_0_stride{i}", typ=int, const=True, memory="host") for i in range(ndim)]
        stride_params += [self.kern_param_ident(f"src_1_stride{i}", typ=int, const=True, memory="host") for i in range(ndim)]
        stride_params += [self.kern_param_ident(f"dst_stride{i}", typ=int, const=True, memory="host") for i in range(ndim)]
        params = (
            self.kern_param_ident("src_0_offset", typ=int, pointer=False, const=True, memory="host"),
            self.kern_param_ident("src_1_offset", typ=int, pointer=False, const=True, memory="host"),
            self.kern_param_ident("dst_offset", typ=int, pointer=False, const=True, memory="host"),
            *stride_params,
            self.kern_param_ident("src_0", typ=float, pointer=True, const=True, memory="device"),
            self.kern_param_ident("src_1", typ=float, pointer=True, const=True, memory="device"),
            self.kern_param_ident("dst", typ=float, pointer=True, const=True, memory="device"),
        )
        return self.generate_kernel(code, ndim, 2, params)

    def generate_kernel(self, code: OpCode, ndim: int, param_cnt: int, param_exprs: tuple[str]):
        indent = self.indent()
        header = self.header(code)
        kern_qual = self.kern_qualifier(code)
        kern_name = to_kern_name(code, ndim)
        kern_body_lines = self.thread_idx_expr(ndim, param_cnt)
        kern_body_lines += self.kern_body(code, ndim)
        return kern_name, f"{kern_qual} {kern_name}({', '.join(param_exprs)}) {{\n{indent}{f"\n{indent}".join(kern_body_lines)}\n}}"


class KernelManager:
    def __init__(self):
        self.kerns: list[Kernel] = []

    def load(self, kerns: Kernel):
        # kern cache
        new_kerns = [kern for kern in kerns if kern.name not in [k.name for k in self.kerns]]
        if new_kerns:
            fps = self.load_kern_ptr(new_kerns)
            self.kerns.extend([Kernel(nk.name, nk.src, fp) for nk, fp in zip(new_kerns, fps)])

    def get_kern(self, name):
        for k in self.kerns:
            if k.name == name:
                return k

    def load_kern_ptr(self, *params):
        raise NotImplementedError()

    def invoke(self, kern_name, grid, block, params):
        raise NotImplementedError()
