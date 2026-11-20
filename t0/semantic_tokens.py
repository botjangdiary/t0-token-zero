"""Dedicated semantic vocabulary for *hypothetical* T0-native LLM decoding.

These are token IDs, not tokens of GPT-6 or another existing model. To make them
actual single output tokens, the serving model needs a matching vocabulary and
training. The integer stream round-trips into the same verified IR.
"""
from __future__ import annotations
import struct

from .ir import Program, Node, Ref, OPCODES, REV_OPCODES, OPS, validate

EOS = 1
RET_LAST = 3
RET_REF = 4
FN_EXT = 5  # omitted means F0
SIG_BASE = 16  # arity (1..8) x return type x FP mode => 32 single tokens
OP_BASE = 64
ARG_BASE = 128  # @0 through @7
TMP_BASE = 160  # %0 through %63
TMP_EXT = 224  # 16-bit temporary index as two raw-byte tokens
IMM_ZERO, IMM_ONE, IMM_NEGONE, IMM_HALF, IMM_RAW = 240, 241, 242, 243, 244
BYTE_BASE = 512  # byte b maps to one 512+b token


def _byte_tokens(raw: bytes) -> list[int]:
    return [BYTE_BASE + b for b in raw]


def _encode_ref(ref: Ref) -> list[int]:
    if ref.kind == "arg":
        if not 0 <= ref.index <= 7:
            raise ValueError("invalid argument reference")
        return [ARG_BASE + ref.index]
    if ref.kind == "tmp":
        if ref.index < 64:
            return [TMP_BASE + ref.index]
        if ref.index > 65535:
            raise ValueError("temporary index exceeds vocabulary extension")
        return [TMP_EXT, *_byte_tokens(ref.index.to_bytes(2, "little"))]
    if ref.kind == "imm":
        raw = struct.pack("<f", ref.value)
        for v, special in ((0.0, IMM_ZERO), (1.0, IMM_ONE), (-1.0, IMM_NEGONE), (0.5, IMM_HALF)):
            if raw == struct.pack("<f", v):
                return [special]
        return [IMM_RAW, *_byte_tokens(raw)]
    raise ValueError("invalid reference kind")


def encode_semantic_tokens(p: Program) -> list[int]:
    """Produce exact semantic-vocabulary IDs; optional F0 and last-result shortcuts."""
    validate(p)
    out = []
    if p.fn_id:
        out += [FN_EXT, *_byte_tokens(p.fn_id.to_bytes(2, "little"))]
    sig = SIG_BASE + (len(p.arg_types) - 1) * 4 + (2 if p.return_type == "u64" else 0) + (1 if p.mode == "fast" else 0)
    out.append(sig)
    for node in p.nodes:
        out.append(OP_BASE + OPCODES[node.op])
        for ref in node.operands:
            out.extend(_encode_ref(ref))
    if p.nodes and p.result == Ref.tmp(len(p.nodes) - 1):
        out.append(RET_LAST)
    else:
        out.append(RET_REF)
        out.extend(_encode_ref(p.result))
    out.append(EOS)
    return out


class _Tokens:
    def __init__(self, tokens: list[int]):
        self.tokens = tokens
        self.pos = 0

    def take(self):
        if self.pos >= len(self.tokens):
            raise ValueError("truncated semantic token stream")
        result = self.tokens[self.pos]
        self.pos += 1
        if not isinstance(result, int) or isinstance(result, bool):
            raise ValueError("tokens must be integers")
        return result

    def raw_bytes(self, n):
        raw = []
        for _ in range(n):
            t = self.take()
            if not BYTE_BASE <= t < BYTE_BASE + 256:
                raise ValueError("expected literal-byte token")
            raw.append(t - BYTE_BASE)
        return bytes(raw)

    def ref(self):
        tok = self.take()
        if ARG_BASE <= tok <= ARG_BASE + 7:
            return Ref.arg(tok - ARG_BASE)
        if TMP_BASE <= tok < TMP_BASE + 64:
            return Ref.tmp(tok - TMP_BASE)
        if tok == TMP_EXT:
            return Ref.tmp(int.from_bytes(self.raw_bytes(2), "little"))
        for special, value in ((IMM_ZERO, 0.0), (IMM_ONE, 1.0), (IMM_NEGONE, -1.0), (IMM_HALF, 0.5)):
            if tok == special:
                return Ref.imm(value)
        if tok == IMM_RAW:
            return Ref.imm(struct.unpack("<f", self.raw_bytes(4))[0])
        raise ValueError(f"invalid reference semantic token: {tok}")


def decode_semantic_tokens(tokens: list[int]) -> Program:
    """Parse discrete tokens and apply independent SSA type verification."""
    if len(tokens) > 100000:
        raise ValueError("semantic token stream too long")
    t = _Tokens(tokens)
    first = t.take()
    fn_id = 0
    if first == FN_EXT:
        fn_id = int.from_bytes(t.raw_bytes(2), "little")
        first = t.take()
    if not SIG_BASE <= first < SIG_BASE + 32:
        raise ValueError("expected signature semantic token")
    sig = first - SIG_BASE
    arity = sig // 4 + 1
    ret = "u64" if sig % 4 >= 2 else "f32"
    mode = "fast" if sig % 2 else "strict"
    nodes = []
    while True:
        opcode = t.take()
        if opcode == RET_LAST:
            if not nodes:
                raise ValueError("RET_LAST requires at least one node")
            result = Ref.tmp(len(nodes) - 1)
            break
        if opcode == RET_REF:
            result = t.ref()
            break
        name = REV_OPCODES.get(opcode - OP_BASE)
        if not name:
            raise ValueError(f"invalid operation semantic token: {opcode}")
        nodes.append(Node(name, tuple(t.ref() for _ in range(OPS[name]))))
        if len(nodes) > 10000:
            raise ValueError("node count limit exceeded")
    if t.take() != EOS or t.pos != len(t.tokens):
        raise ValueError("expected EOS with no trailing tokens")
    p = Program(fn_id, ("f32[*]",) * arity, ret, mode, tuple(nodes), result)
    validate(p)
    return p
