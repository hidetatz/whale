from __future__ import annotations

import typing
from ctypes import c_void_p
from dataclasses import dataclass
from enum import IntEnum, auto


class OpCode(IntEnum):
    RECIP = auto()
    ADD = auto()
    MUL = auto()
    POW = auto()
    LOG = auto()
    COPY = auto()
    SUM = auto()

    def __str__(self):
        return self.name


@dataclass
class Kernel:
    name: str
    src: str
    func_pointer: c_void_p  # on C side


class VTypeCode(IntEnum):
    VOID = auto()
    I32 = auto()
    I64 = auto()
    F32 = auto()


class LangFlavor:
    def indent_size(self) -> int:
        raise NotImplementedError()

    def line_term(self) -> str:
        raise NotImplementedError()

    def kern_qualifier(self) -> str:
        raise NotImplementedError()

    def block_start(self) -> str:
        raise NotImplementedError()

    def block_end(self) -> str:
        raise NotImplementedError()

    def if_cond_start(self) -> str:
        raise NotImplementedError()

    def if_cond_end(self) -> str:
        raise NotImplementedError()

    def typegen(self) -> str:
        raise NotImplementedError()

    def unary_op_gen(self) -> str:
        raise NotImplementedError()

    def grid_dim(self, dim: str) -> str:
        raise NotImplementedError()

    def block_idx(self, dim: str) -> str:
        raise NotImplementedError()

    def block_dim(self, dim: str) -> str:
        raise NotImplementedError()

    def thread_idx(self, dim: str) -> str:
        raise NotImplementedError()


@dataclass
class VType:
    typ: VTypeCode
    pointer: bool = False


class KernelGeneratorBuffer:
    def __init__(self, flavor: LangFlavor):
        self.buff = []
        self.depth = 0
        self.flavor = flavor
        self.indent_size = flavor.indent_size()
        self.line_term = flavor.line_term()
        self.kern_qualifier = flavor.kern_qualifier()
        self.block_start = flavor.block_start()
        self.block_end = flavor.block_end()
        self.if_cond_start = flavor.if_cond_start()
        self.if_cond_end = flavor.if_cond_end()
        self.loop_cond_start = flavor.loop_cond_start()
        self.loop_cond_end = flavor.loop_cond_end()
        self.loop_cond = flavor.loop_cond
        self.typegen = flavor.typegen
        self.unary_op_gen = flavor.unary_op_gen
        self.binary_op_gen = flavor.binary_op_gen
        self.grid_dim = flavor.grid_dim
        self.block_idx = flavor.block_idx
        self.block_dim = flavor.block_dim
        self.thread_idx = flavor.thread_idx

    def indent(self) -> str:
        return " " * self.indent_size * self.depth

    def addline(self, expr: str, term: bool = True) -> None:
        self.buff.append(f"{self.indent()}{expr}{self.line_term if term else ''}")

    def to_str(self):
        return "\n".join(self.buff)

    #
    # statements
    #

    def kernel_start(self, name: str, ret: VType, params: list[tuple[VType, str]]) -> None:
        ps = ", ".join(f"{self.type_expr(p[0])} {p[1]}" for p in params)
        self.addline(f"{self.kern_qualifier} {self.typegen(ret)} {name} ({ps}){self.block_start}", term=False)
        self.depth += 1

    def kernel_end(self) -> None:
        self.depth -= 1
        self.addline(f"{self.block_end}", term=False)

    def init(self, typ: VType, name: str, right: str) -> None:
        self.addline(f"{self.type_expr(typ)} {name} = {right}")

    def assign(self, left: str, right: str) -> None:
        self.addline(f"{left} = {right}")

    def if_start(self, cond: str) -> None:
        self.addline(f"if {self.if_cond_start}{cond}{self.if_cond_end} {self.block_start}", term=False)
        self.depth += 1

    def if_end(self) -> None:
        self.depth -= 1
        self.addline(f"{self.block_end}", term=False)

    def loop_start(self, start: str, stop: str, var: str = "i") -> None:
        self.addline(f"for {self.loop_cond_start}{self.loop_cond(start, stop, var)}{self.loop_cond_end} {self.block_start}", term=False)
        self.depth += 1

    def loop_end(self) -> None:
        self.depth -= 1
        self.addline(f"{self.block_end}", term=False)

    #
    # expressions
    #

    def type_expr(self, typ: VType) -> str:
        return self.typegen(typ)

    def binary_multi_expr(self, operands: list[str], operator: str) -> str:
        return "(" + f" {operator} ".join(operands) + ")"

    def unary_op_expr(self, operand: str, valid_operand: str, op: Opcode) -> str:
        return "(" + self.unary_op_gen(operand, valid_operand, op) + ")"

    def binary_op_expr(self, loperand: str, valid_loperand: str, roperand: str, valid_roperand: str, op: Opcode) -> str:
        return "(" + self.binary_op_gen(loperand, valid_loperand, roperand, valid_roperand, op) + ")"

    def binary_expr(self, left: str, operator: str, right: str) -> str:
        return "(" + f"{left} {operator} {right}" + ")"

    def binary_expr_from_opcode(self, code: OpCode, left: str, right: str) -> str:
        if code == OpCode.SUM:
            return f"{left} + {right}"

        raise NotImplementedError()

    def ternary_expr(self, cond: str, left: str, right: str) -> str:
        return f"{cond} ? {left} : {right}"  # This does not work for Python flavor

    def cast_expr(self, typ: VType, target: str) -> str:
        return f"({self.type_expr(typ)}){target}"

    def index_expr(self, arr: str, index: str) -> str:
        return f"{arr}[{index}]"


def to_kern_name(code: OpCode, dim: int) -> str:
    return f"{code.__str__().lower()}_dim{dim}"


def to_reduce_kern_name(code: OpCode, dim: int, axis: int) -> str:
    return f"{code.__str__().lower()}_dim{dim}_axis{axis}"


class CodeGenerator:
    def __init__(self, flavor):
        self.flavor = flavor

    # not really optimized, but simple and good for the baseline
    def _init_linear_idx_2dim_blocks_1dim_threads(self, buff: KernelGeneratorBuffer) -> None:
        # i64 block_idx = (i64)blockIdx.y * gridDim.x + blockIdx.x;
        # i64 idx = block_idx * blockDim.x + threadIdx.x;
        block_idx = buff.binary_expr(
            buff.binary_expr(buff.cast_expr(VType(VTypeCode.I64), buff.block_idx("y")), "*", buff.grid_dim("x")), "+", buff.block_idx("x")
        )
        buff.init(VType(VTypeCode.I64), "block_pos", block_idx)
        idx = buff.binary_expr(buff.binary_expr("block_pos", "*", buff.block_dim("x")), "+", buff.thread_idx("x"))
        buff.init(VType(VTypeCode.I64), "idx", idx)

    def generate_unary_kernel(self, code: OpCode, ndim: int) -> str:
        buff = KernelGeneratorBuffer(self.flavor)

        kern_name = to_kern_name(code, ndim)
        buff.kernel_start(
            kern_name,
            VType(VTypeCode.VOID),
            [
                (VType(VTypeCode.I32, False), "total"),
                *[(VType(VTypeCode.I32, False), f"dst_shape_{i}") for i in range(ndim)],
                *[(VType(VTypeCode.I32, False), f"src_0_stride{i}") for i in range(ndim)],
                *[(VType(VTypeCode.I32, False), f"src_0_valid_area_{i}_{suffix}") for i in range(ndim) for suffix in ["start", "end"]],
                (VType(VTypeCode.I32, False), "src_0_offset"),
                (VType(VTypeCode.F32, True), "src_0"),
                (VType(VTypeCode.F32, True), "dst"),
            ],
        )

        # calculate linearized thread index
        self._init_linear_idx_2dim_blocks_1dim_threads(buff)
        buff.if_start(f"idx < total")

        # restore dimensions index
        if ndim != 0:
            buff.init(VType(VTypeCode.I64), "remaining", "idx")
            for i in range(ndim - 1, -1, -1):
                buff.init(VType(VTypeCode.I64), f"src_idx_{i}", buff.binary_expr("remaining", "%", f"dst_shape_{i}"))
                if i != 0:
                    buff.assign("remaining", buff.binary_expr("remaining", "/", f"dst_shape_{i}"))

        # actual index calculation
        if ndim == 0:
            buff.init(VType(VTypeCode.I32), "src_0_lidx", "0")
        else:
            exprs = [buff.binary_expr(f"src_idx_{i}", "*", f"src_0_stride{i}") for i in range(ndim)]
            buff.init(VType(VTypeCode.I32), "src_0_lidx", buff.binary_expr("src_0_offset", "+", buff.binary_multi_expr(exprs, "+")))

        # valid_area check
        if ndim == 0:
            buff.init(VType(VTypeCode.I32), "src_0_lidx_valid", "1")
        else:
            exprs = [
                buff.binary_expr(
                    buff.binary_expr(f"src_0_valid_area_{i}_start", "<=", f"src_idx_{i}"),
                    "&&",
                    buff.binary_expr(f"src_idx_{i}", "<", f"src_0_valid_area_{i}_end"),
                )
                for i in range(ndim)
            ]
            buff.init(VType(VTypeCode.I32), "src_0_lidx_valid", buff.binary_multi_expr(exprs, "&&"))

        buff.init(VType(VTypeCode.F32), "src_0_val", buff.ternary_expr("src_0_lidx_valid", buff.index_expr("src_0", "src_0_lidx"), "0.0f"))
        buff.assign(buff.index_expr("dst", "idx"), buff.unary_op_expr("src_0_val", "src_0_lidx_valid", code))

        buff.if_end()
        buff.kernel_end()
        return kern_name, buff.to_str()

    def generate_binary_kernel(self, code: OpCode, ndim: int):
        buff = KernelGeneratorBuffer(self.flavor)

        kern_name = to_kern_name(code, ndim)
        buff.kernel_start(
            kern_name,
            VType(VTypeCode.VOID),
            [
                (VType(VTypeCode.I32, False), "total"),
                *[(VType(VTypeCode.I32, False), f"dst_shape_{i}") for i in range(ndim)],
                *[(VType(VTypeCode.I32, False), f"src_0_stride{i}") for i in range(ndim)],
                *[(VType(VTypeCode.I32, False), f"src_1_stride{i}") for i in range(ndim)],
                *[(VType(VTypeCode.I32, False), f"src_0_valid_area_{i}_{suffix}") for i in range(ndim) for suffix in ["start", "end"]],
                *[(VType(VTypeCode.I32, False), f"src_1_valid_area_{i}_{suffix}") for i in range(ndim) for suffix in ["start", "end"]],
                (VType(VTypeCode.I32, False), "src_0_offset"),
                (VType(VTypeCode.I32, False), "src_1_offset"),
                (VType(VTypeCode.F32, True), "src_0"),
                (VType(VTypeCode.F32, True), "src_1"),
                (VType(VTypeCode.F32, True), "dst"),
            ],
        )

        # calculate linearized thread index
        self._init_linear_idx_2dim_blocks_1dim_threads(buff)
        buff.if_start(f"idx < total")

        # restore dimensions index
        if ndim != 0:
            buff.init(VType(VTypeCode.I64), "remaining", "idx")
            for i in range(ndim - 1, -1, -1):
                buff.init(VType(VTypeCode.I64), f"src_idx_{i}", buff.binary_expr("remaining", "%", f"dst_shape_{i}"))
                if i != 0:
                    buff.assign("remaining", buff.binary_expr("remaining", "/", f"dst_shape_{i}"))

        # actual index calculation
        if ndim == 0:
            buff.init(VType(VTypeCode.I32), "src_0_lidx", "0")
            buff.init(VType(VTypeCode.I32), "src_1_lidx", "0")
        else:
            exprs0 = [buff.binary_expr(f"src_idx_{i}", "*", f"src_0_stride{i}") for i in range(ndim)]
            buff.init(VType(VTypeCode.I32), "src_0_lidx", buff.binary_expr("src_0_offset", "+", buff.binary_multi_expr(exprs0, "+")))
            exprs1 = [buff.binary_expr(f"src_idx_{i}", "*", f"src_1_stride{i}") for i in range(ndim)]
            buff.init(VType(VTypeCode.I32), "src_1_lidx", buff.binary_expr("src_1_offset", "+", buff.binary_multi_expr(exprs1, "+")))

        # valid_area check
        if ndim == 0:
            buff.init(VType(VTypeCode.I32), "src_0_lidx_valid", "1")
            buff.init(VType(VTypeCode.I32), "src_1_lidx_valid", "1")
        else:
            exprs0 = [
                buff.binary_expr(
                    buff.binary_expr(f"src_0_valid_area_{i}_start", "<=", f"src_idx_{i}"),
                    "&&",
                    buff.binary_expr(f"src_idx_{i}", "<", f"src_0_valid_area_{i}_end"),
                )
                for i in range(ndim)
            ]
            buff.init(VType(VTypeCode.I32), "src_0_lidx_valid", buff.binary_multi_expr(exprs0, "&&"))
            exprs1 = [
                buff.binary_expr(
                    buff.binary_expr(f"src_1_valid_area_{i}_start", "<=", f"src_idx_{i}"),
                    "&&",
                    buff.binary_expr(f"src_idx_{i}", "<", f"src_1_valid_area_{i}_end"),
                )
                for i in range(ndim)
            ]
            buff.init(VType(VTypeCode.I32), "src_1_lidx_valid", buff.binary_multi_expr(exprs1, "&&"))

        buff.init(VType(VTypeCode.F32), "src_0_val", buff.ternary_expr("src_0_lidx_valid", buff.index_expr("src_0", "src_0_lidx"), "0.0f"))
        buff.init(VType(VTypeCode.F32), "src_1_val", buff.ternary_expr("src_1_lidx_valid", buff.index_expr("src_1", "src_1_lidx"), "0.0f"))
        buff.assign(buff.index_expr("dst", "idx"), buff.binary_op_expr("src_0_val", "src_0_lidx_valid", "src_1_val", "src_1_lidx_valid", code))

        buff.if_end()
        buff.kernel_end()
        return kern_name, buff.to_str()

    # reduces by the single axis, the output size threads will be invoked. Must be optimized
    def generate_reduce_kernel(self, code: OpCode, ndim: int, axis: int):
        buff = KernelGeneratorBuffer(self.flavor)

        kern_name = to_reduce_kern_name(code, ndim, axis)
        buff.kernel_start(
            kern_name,
            VType(VTypeCode.VOID),
            [
                (VType(VTypeCode.I32), "total"),
                (VType(VTypeCode.I32), f"dim{axis}"),
                *[(VType(VTypeCode.I32), f"dst_shape_{i}") for i in range(ndim)],
                *[(VType(VTypeCode.I32), f"src_0_stride{i}") for i in range(ndim)],
                *[(VType(VTypeCode.I32), f"src_0_valid_area_{i}_{suffix}") for i in range(ndim) for suffix in ["start", "end"]],
                (VType(VTypeCode.I32), "src_0_offset"),
                (VType(VTypeCode.F32, True), "src_0"),
                (VType(VTypeCode.F32, True), "dst"),
            ],
        )

        # calculate linearized thread index
        self._init_linear_idx_2dim_blocks_1dim_threads(buff)
        buff.if_start(f"idx < total")

        # restore dimensions index
        if 1 < ndim:
            buff.init(VType(VTypeCode.I64), "remaining", "idx")
            rest_axes = list(range(ndim - 1, -1, -1))
            rest_axes.remove(axis)
            for i, a in enumerate(rest_axes):
                buff.init(VType(VTypeCode.I64), f"src_idx_{a}", buff.binary_expr("remaining", "%", f"dst_shape_{a}"))
                if i != len(rest_axes) - 1:
                    buff.assign("remaining", buff.binary_expr("remaining", "/", f"dst_shape_{a}"))

        acc = "0.0f" if code == OpCode.SUM else "0.0f"

        buff.init(VType(VTypeCode.F32), "acc", acc)
        buff.loop_start("0", f"dim{axis}", var=f"src_idx_{axis}")

        # actual index calculation
        if ndim == 0:
            buff.init(VType(VTypeCode.I32), "src_0_lidx", "0")
        else:
            exprs = [buff.binary_expr(f"src_idx_{i}", "*", f"src_0_stride{i}") for i in range(ndim)]
            buff.init(VType(VTypeCode.I32), "src_0_lidx", buff.binary_expr("src_0_offset", "+", buff.binary_multi_expr(exprs, "+")))

        # valid_area check
        if ndim == 0:
            buff.init(VType(VTypeCode.I32), "src_0_lidx_valid", "1")
        else:
            exprs = [
                buff.binary_expr(
                    buff.binary_expr(f"src_0_valid_area_{i}_start", "<=", f"src_idx_{i}"),
                    "&&",
                    buff.binary_expr(f"src_idx_{i}", "<", f"src_0_valid_area_{i}_end"),
                )
                for i in range(ndim)
            ]
            buff.init(VType(VTypeCode.I32), "src_0_lidx_valid", buff.binary_multi_expr(exprs, "&&"))

        buff.init(VType(VTypeCode.F32), "src_0_val", buff.ternary_expr("src_0_lidx_valid", buff.index_expr("src_0", "src_0_lidx"), "0.0f"))

        buff.assign("acc", buff.binary_expr_from_opcode(code, "acc", "src_0_val"))

        buff.loop_end()
        buff.assign(buff.index_expr("dst", "idx"), "acc")

        buff.if_end()
        buff.kernel_end()
        return kern_name, buff.to_str()


class KernelManager:
    def __init__(self):
        self.kerns: list[Kernel] = []

    def load(self, kerns: list[Kernel]) -> None:
        # kern cache
        new_kerns = [kern for kern in kerns if kern.name not in [k.name for k in self.kerns]]
        if new_kerns:
            fps = self.load_kern_ptr(new_kerns)
            self.kerns.extend([Kernel(nk.name, nk.src, fp) for nk, fp in zip(new_kerns, fps)])
        self.kerns.extend(new_kerns)

    def get_kern(self, name) -> Kernel | None:
        for k in self.kerns:
            if k.name == name:
                return k
        return None

    def load_kern_ptr(self, kerns: list[Kernel]) -> list[c_void_p]:
        raise NotImplementedError()

    def invoke(self, kern_name: str, grid: int | tuple[int], block: int | tuple[int], params: tuple[typing.Any]):
        raise NotImplementedError()
