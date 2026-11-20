"""Slow but simple f32-correct reference executor for validating native output."""
from __future__ import annotations
import math
from typing import Sequence
from .ir import Program, Ref, f32, validate
from .backend import normalized_arrays


def run_reference(program: Program, *inputs: Sequence[float]):
    p = program
    validate(p)
    arrays = normalized_arrays(p, inputs)
    n = len(arrays[0])
    vals: list[object] = []

    def get(ref: Ref):
        if ref.kind == "arg":
            return arrays[ref.index]
        if ref.kind == "tmp":
            return vals[ref.index]
        return ref.value

    def vec2(a, b, op, cast=False):
        if isinstance(a, list) or isinstance(b, list):
            return [f32(op(a[i] if isinstance(a, list) else a,
                           b[i] if isinstance(b, list) else b)) if cast else
                    op(a[i] if isinstance(a, list) else a,
                       b[i] if isinstance(b, list) else b) for i in range(n)]
        v = op(a, b)
        return f32(v) if cast else v

    def reduce_sum(items):
        accum = 0.0
        for value in items:
            accum = f32(accum + value)
        return accum

    for node in p.nodes:
        op = node.op
        a = [get(ref) for ref in node.operands]
        if op == "gt":
            result = vec2(a[0], a[1], lambda x, y: x > y)
        elif op == "mul":
            result = vec2(a[0], a[1], lambda x, y: x * y, cast=True)
        elif op == "sel":
            if any(isinstance(v, list) for v in a):
                result = [(a[1][i] if isinstance(a[1], list) else a[1]) if
                          (a[0][i] if isinstance(a[0], list) else a[0]) else
                          (a[2][i] if isinstance(a[2], list) else a[2])
                          for i in range(n)]
            else:
                result = a[1] if a[0] else a[2]
        elif op == "sum":
            result = reduce_sum(a[0])
        elif op == "rp2":
            result = reduce_sum(f32(x * x) if x > 0 else 0.0 for x in a[0])
        elif op == "dot":
            result = reduce_sum(f32(x * y) for x, y in zip(a[0], a[1]))
        elif op == "countp":
            result = sum(x > 0 for x in a[0])
        elif op == "max":
            result = float('nan') if any(math.isnan(v) for v in a[0]) else max(a[0])
        elif op == "mean":
            result = f32(reduce_sum(a[0]) / n)
        else:
            raise AssertionError(op)
        vals.append(result)
    return get(p.result)
