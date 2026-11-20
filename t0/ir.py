"""Typed, side-effect-free SSA IR for the T0 prototype.

T0 is deliberately small. It has no pointers, user-defined functions, control
flow, filesystem, or runtime code evaluation. The IR is validated before use.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import struct


ARG_TYPES = {"f32[*]"}
RETURN_TYPES = {"f32", "u64"}
OPS = {
    "gt": 2,
    "mul": 2,
    "sel": 3,
    "sum": 1,
    "rp2": 1,
    "dot": 2,
    "countp": 1,
    "max": 1,
    "mean": 1,
}
OPCODES = {"gt": 1, "mul": 2, "sel": 3, "sum": 4,
           "rp2": 5, "dot": 6, "countp": 7, "max": 8, "mean": 9}
REV_OPCODES = {v: k for k, v in OPCODES.items()}


def f32(n: float) -> float:
    """Round an immediate/element to the f32 format understood by the native ABI."""
    try:
        value = struct.unpack("<f", struct.pack("<f", float(n)))[0]
    except OverflowError as exc:
        raise ValueError("f32 value out of range") from exc
    return value


@dataclass(frozen=True)
class Ref:
    kind: str  # arg, tmp, imm
    index: int = 0
    value: float = 0.0

    @staticmethod
    def arg(index: int) -> "Ref":
        return Ref("arg", index)

    @staticmethod
    def tmp(index: int) -> "Ref":
        return Ref("tmp", index)

    @staticmethod
    def imm(value: float) -> "Ref":
        return Ref("imm", 0, f32(value))


@dataclass(frozen=True)
class Node:
    op: str
    operands: tuple[Ref, ...]


@dataclass(frozen=True)
class Program:
    fn_id: int
    arg_types: tuple[str, ...]
    return_type: str
    mode: str
    nodes: tuple[Node, ...]
    result: Ref


def infer_op_type(op: str, types: tuple[str, ...]) -> str:
    if op not in OPS or len(types) != OPS[op]:
        raise ValueError(f"unknown op or operand count: {op}/{len(types)}")
    if op in {"gt", "mul"}:
        if not all(t in {"f32", "f32[*]"} for t in types):
            raise ValueError(f"{op} accepts only f32 or f32[*]")
        out_vec = "f32[*]" in types
        if op == "gt":
            return "bool[*]" if out_vec else "bool"
        return "f32[*]" if out_vec else "f32"
    if op == "sel":
        c, a, b = types
        if c not in {"bool", "bool[*]"} or a not in {"f32", "f32[*]"} or b not in {"f32", "f32[*]"}:
            raise ValueError("sel expects (bool, f32, f32), with optional broadcasting")
        # A vector-valued conditional can select scalars and vice versa.
        return "f32[*]" if "[*]" in (c + a + b) else "f32"
    if op in {"sum", "rp2", "countp", "max", "mean"}:
        if types != ("f32[*]",):
            raise ValueError(f"{op} expects one f32[*]")
        return "u64" if op == "countp" else "f32"
    if op == "dot":
        if types != ("f32[*]", "f32[*]"):
            raise ValueError("dot expects two f32[*]")
        return "f32"
    raise AssertionError(op)


def validate(p: Program) -> tuple[str, ...]:
    """Typecheck every node; return per-node inferred types.

    No forward references, out-of-range arguments, or impurity are permitted.
    """
    if not (0 <= p.fn_id <= 65535):
        raise ValueError("function ID out of range")
    if p.mode not in {"strict", "fast"}:
        raise ValueError("mode must be strict or fast")
    if not (1 <= len(p.arg_types) <= 8) or any(t not in ARG_TYPES for t in p.arg_types):
        raise ValueError("prototype accepts 1-8 f32[*] arguments")
    if p.return_type not in RETURN_TYPES:
        raise ValueError("unsupported return type")
    if len(p.nodes) > 10000:
        raise ValueError("too many IR nodes")

    inferred: list[str] = []

    def ref_type(r: Ref) -> str:
        if r.kind == "arg" and 0 <= r.index < len(p.arg_types):
            return p.arg_types[r.index]
        if r.kind == "tmp" and 0 <= r.index < len(inferred):
            return inferred[r.index]
        if r.kind == "imm" and math.isfinite(r.value) and r.value == f32(r.value):
            return "f32"
        raise ValueError(f"invalid, non-finite, or forward reference: {r}")

    for i, node in enumerate(p.nodes):
        inferred.append(infer_op_type(node.op, tuple(ref_type(r) for r in node.operands)))
    result_type = ref_type(p.result)
    if result_type != p.return_type:
        raise ValueError(f"return type mismatch: declared {p.return_type}, got {result_type}")
    return tuple(inferred)


def pretty_ref(r: Ref) -> str:
    if r.kind == "arg":
        return f"@{r.index}"
    if r.kind == "tmp":
        return f"%{r.index}"
    if r.kind == "imm":
        s = format(r.value, ".9g")
        if "." not in s and "e" not in s:
            s += ".0"
        return s + "f"
    raise ValueError(f"bad ref kind: {r.kind}")


def format_program(p: Program, compact: bool = False) -> str:
    validate(p)
    head = f"F{p.fn_id}({','.join(p.arg_types)})->{p.return_type}!p~{p.mode}{{"
    stmts = [f"%{i}={node.op}({','.join(pretty_ref(r) for r in node.operands)})"
             for i, node in enumerate(p.nodes)]
    stmts.append(f"R={pretty_ref(p.result)}")
    if compact:
        return head + ";".join(stmts) + "}"
    return head + "\n  " + "\n  ".join(stmts) + "\n}"
