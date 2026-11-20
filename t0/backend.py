"""T0 native backend: generate C11, then GCC/Clang -O3 shared library."""
from __future__ import annotations

import ctypes
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Sequence

from .ir import Program, Ref, validate, f32
from .optimizer import optimize


def _literal(value: float) -> str:
    s = format(f32(value), ".9g")
    if "." not in s and "e" not in s:
        s += ".0"
    return s + "f"


def generate_c(program: Program, optimize_first: bool = True) -> str:
    """Compile well-typed SSA to f32 C. Vector producers fuse into consumers.

    Scalar-producing nodes are evaluated in SSA order. Vector nodes are lazily
    inlined at the element level; no intermediate array allocations.
    """
    p = optimize(program) if optimize_first else program
    inferred = validate(p)
    args = ", ".join(f"const float *a{i}" for i in range(len(p.arg_types)))
    result_c_type = "uint64_t" if p.return_type == "u64" else "float"
    lines = [
        "/* Generated from T0; no runtime allocations. */",
        "#include <stdint.h>",
        "#include <stddef.h>",
        "#include <math.h>",
        f"{result_c_type} t0_run({args}, size_t n) {{",
    ]

    def expression(ref: Ref, idx: str = "i") -> str:
        if ref.kind == "arg":
            return f"a{ref.index}[{idx}]"
        if ref.kind == "imm":
            return _literal(ref.value)
        if ref.kind != "tmp":
            raise ValueError(f"bad reference {ref}")
        if "[*]" not in inferred[ref.index]:
            return f"t{ref.index}"
        node = p.nodes[ref.index]
        args_expr = [expression(r, idx) for r in node.operands]
        if node.op == "mul":
            return f"({args_expr[0]} * {args_expr[1]})"
        if node.op == "gt":
            return f"({args_expr[0]} > {args_expr[1]})"
        if node.op == "sel":
            return f"({args_expr[0]} ? {args_expr[1]} : {args_expr[2]})"
        raise ValueError(f"unexpected vector op: {node.op}")

    for i, node in enumerate(p.nodes):
        if "[*]" in inferred[i]:
            continue
        op = node.op
        if op in {"mul", "gt", "sel"}:
            a = [expression(r) for r in node.operands]
            expr = (f"({a[0]} * {a[1]})" if op == "mul"
                    else f"({a[0]} > {a[1]})" if op == "gt"
                    else f"({a[0]} ? {a[1]} : {a[2]})")
            t = "_Bool" if inferred[i] == "bool" else "float"
            lines.append(f"  {t} t{i} = {expr};")
        elif op in {"sum", "dot", "rp2", "countp", "mean", "max"}:
            # A reduction is a fusion boundary. Everything upstream that is
            # element-wise is evaluated inside this single loop.
            if op == "countp":
                lines += [f"  uint64_t t{i} = 0;",
                          "  for (size_t i = 0; i < n; ++i) {",
                          f"    if ({expression(node.operands[0])} > 0.0f) ++t{i};",
                          "  }"]
            elif op == "max":
                lines += [f"  float t{i} = NAN;",
                          "  if (n != 0) {",
                          f"    t{i} = {expression(node.operands[0], '0')};",
                          "    for (size_t i = 1; i < n; ++i) {",
                          f"      float x = {expression(node.operands[0])};",
                          f"      if (isnan(x) || isnan(t{i})) {{ t{i} = NAN; break; }}",
                          f"      if (x > t{i}) t{i} = x;",
                          "    }",
                          "  }"]
            elif op == "rp2":
                expr = expression(node.operands[0])
                lines += [f"  float t{i} = 0.0f;",
                          "  for (size_t i = 0; i < n; ++i) {",
                          f"    float x = {expr};",
                          f"    if (x > 0.0f) t{i} += x * x;",
                          "  }"]
            else:
                if op == "sum" or op == "mean":
                    value = expression(node.operands[0])
                elif op == "dot":
                    value = f"({expression(node.operands[0])} * {expression(node.operands[1])})"
                else:
                    raise AssertionError(op)
                lines += [f"  float t{i} = 0.0f;",
                          "  for (size_t i = 0; i < n; ++i) {",
                          f"    t{i} += {value};",
                          "  }"]
                if op == "mean":
                    lines.append(f"  t{i} = n ? (t{i} / (float)n) : NAN;")
        else:
            raise ValueError(f"unsupported C backend operation: {op}")

    if p.result.kind == "tmp":
        value = f"t{p.result.index}"
    elif p.result.kind == "imm":
        value = _literal(p.result.value)
    else:
        raise ValueError("cannot return an entire array from a scalar function")
    lines += [f"  return {value};", "}", ""]
    return "\n".join(lines)


def compiler_executable() -> str:
    exe = os.environ.get("CC", "gcc")
    if not shutil.which(exe):
        raise RuntimeError(f"C compiler not found: {exe}; install GCC or Clang, or set CC")
    return exe


def compile_native(program: Program, output: str | Path, *, emit_c: str | Path | None = None,
                   optimize_first: bool = True) -> Path:
    p = optimize(program) if optimize_first else program
    code = generate_c(p, optimize_first=False)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if emit_c is not None:
        Path(emit_c).write_text(code, encoding="utf-8")
    with tempfile.TemporaryDirectory(prefix="t0cc-") as scratch:
        source = Path(scratch) / "kernel.c"
        source.write_text(code, encoding="utf-8")
        flags = ["-O3", "-fPIC", "-shared", "-std=c11", "-ffp-contract=off"]
        if p.mode == "fast":
            flags += ["-ffast-math"]
        else:
            flags += ["-fno-fast-math"]
        cmd = [compiler_executable(), *flags, str(source), "-o", str(output)]
        proc = subprocess.run(cmd, text=True, capture_output=True)
        if proc.returncode:
            raise RuntimeError(f"native compile error: {proc.stderr}")
    return output


def normalized_arrays(program: Program, arrays: Sequence[Sequence[float]]) -> list[list[float]]:
    validate(program)
    if len(arrays) != len(program.arg_types):
        raise ValueError(f"expected {len(program.arg_types)} input arrays, received {len(arrays)}")
    out = [[f32(v) for v in arg] for arg in arrays]
    if len({len(arg) for arg in out}) != 1:
        raise ValueError("all arrays must have the same length")
    if not out[0] and program_uses_nonempty_reduction(program):
        raise ValueError("max/mean requires nonempty input")
    return out


def program_uses_nonempty_reduction(program: Program) -> bool:
    return any(n.op in {"max", "mean"} for n in program.nodes)


class NativeKernel:
    """Compiled library plus stable ctypes arrays; compile time excluded from run timing."""

    def __init__(self, program: Program, shared_library: str | Path):
        self.program = program
        self.lib = ctypes.CDLL(str(Path(shared_library).resolve()))
        self.fn = self.lib.t0_run
        self.fn.argtypes = [ctypes.POINTER(ctypes.c_float)] * len(program.arg_types) + [ctypes.c_size_t]
        self.fn.restype = ctypes.c_uint64 if program.return_type == "u64" else ctypes.c_float

    def prepare(self, *arrays: Sequence[float]):
        checked = normalized_arrays(self.program, arrays)
        c_arrays = [(ctypes.c_float * len(arg))(*arg) for arg in checked]
        n = len(checked[0])
        return c_arrays, n

    def execute_prepared(self, prepared):
        c_arrays, n = prepared
        return self.fn(*c_arrays, n)

    def __call__(self, *arrays: Sequence[float]):
        return self.execute_prepared(self.prepare(*arrays))


def run_native(program: Program, *arrays: Sequence[float]):
    with tempfile.TemporaryDirectory(prefix="t0run-") as scratch:
        path = compile_native(program, Path(scratch) / "kernel.so")
        return NativeKernel(program, path)(*arrays)
