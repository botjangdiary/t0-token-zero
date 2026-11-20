"""Semantics-preserving graph rewrites for T0's initial operation set."""
from __future__ import annotations
from .ir import Program, Node, Ref, validate


def _node(p: Program, r: Ref, op: str | None = None) -> Node | None:
    if r.kind != "tmp" or r.index >= len(p.nodes):
        return None
    node = p.nodes[r.index]
    return node if op is None or node.op == op else None


def _zero(r: Ref) -> bool:
    return r.kind == "imm" and r.value == 0.0


def optimize(p: Program) -> Program:
    """Collapse common producer graphs into single semantic ops.

    A minimal, conservative pass: only rewrites a whole pure expression ending
    in a reduction. An independent IR verifier is run before/after rewrites.
    """
    validate(p)
    root = _node(p, p.result)
    if root is None or root.op != "sum":
        return p
    reduced = root.operands[0]
    product = _node(p, reduced, "mul")
    if product is not None and product.operands[0].kind == "arg" and product.operands[1].kind == "arg":
        a, b = product.operands
        if a != b:
            q = Program(p.fn_id, p.arg_types, p.return_type, p.mode,
                        (Node("dot", (a, b)),), Ref.tmp(0))
            validate(q)
            return q
    selected = _node(p, reduced, "sel")
    if selected is None:
        return p
    cond, yes, no = selected.operands
    test = _node(p, cond, "gt")
    square = _node(p, yes, "mul")
    if test is None or square is None or not _zero(no):
        return p
    x, threshold = test.operands
    if x.kind != "arg" or not _zero(threshold):
        return p
    if square.operands != (x, x):
        return p
    q = Program(p.fn_id, p.arg_types, p.return_type, p.mode,
                (Node("rp2", (x,)),), Ref.tmp(0))
    validate(q)
    return q
