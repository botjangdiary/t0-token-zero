"""Compact, deterministic T0B1 binary container (little endian, CRC32)."""
from __future__ import annotations
import io
import struct
import zlib
from .ir import Program, Node, Ref, OPCODES, REV_OPCODES, validate

MAGIC = b"T0B\x00"
VERSION = 1
MODE_IDS = {"strict": 0, "fast": 1}
TYPE_IDS = {"f32[*]": 1, "f32": 2, "u64": 3}
REVERSE_TYPE_IDS = {v: k for k, v in TYPE_IDS.items()}


def _varint(value: int) -> bytes:
    if not 0 <= value <= 0xFFFFFFFF:
        raise ValueError("varint out of range")
    out = bytearray()
    while value >= 0x80:
        out.append((value & 0x7f) | 0x80)
        value >>= 7
    out.append(value)
    return bytes(out)


def _ref_bytes(r: Ref) -> bytes:
    if r.kind == "arg":
        return _varint((r.index << 2) | 0)
    if r.kind == "tmp":
        return _varint((r.index << 2) | 1)
    if r.kind == "imm":
        return _varint(2) + struct.pack("<f", r.value)
    raise ValueError("invalid ref")


def encode(program: Program) -> bytes:
    validate(program)
    p = program
    out = bytearray(MAGIC)
    out += bytes((VERSION, MODE_IDS[p.mode]))
    out += _varint(p.fn_id)
    out += _varint(len(p.arg_types))
    out += bytes(TYPE_IDS[t] for t in p.arg_types)
    out += bytes((TYPE_IDS[p.return_type],))
    out += _varint(len(p.nodes))
    for n in p.nodes:
        out.append(OPCODES[n.op])
        for r in n.operands:
            out += _ref_bytes(r)
    out += _ref_bytes(p.result)
    out += struct.pack("<I", zlib.crc32(out))
    return bytes(out)


class _Reader:
    def __init__(self, data: bytes):
        self.f = io.BytesIO(data)

    def take(self, n: int) -> bytes:
        val = self.f.read(n)
        if len(val) != n:
            raise ValueError("truncated T0 bytecode")
        return val

    def byte(self) -> int:
        return self.take(1)[0]

    def varint(self) -> int:
        value = 0
        for i in range(5):
            b = self.byte()
            value |= (b & 0x7f) << (7 * i)
            if b < 0x80:
                if value > 0xFFFFFFFF:
                    raise ValueError("varint overflow")
                return value
        raise ValueError("overlong varint")

    def ref(self) -> Ref:
        code = self.varint()
        kind = code & 3
        idx = code >> 2
        if kind == 0:
            return Ref.arg(idx)
        if kind == 1:
            return Ref.tmp(idx)
        if code == 2:
            return Ref.imm(struct.unpack("<f", self.take(4))[0])
        raise ValueError("unknown reference tag")


def decode(data: bytes) -> Program:
    if len(data) < 15 or data[:4] != MAGIC:
        raise ValueError("not a T0B binary")
    if struct.unpack("<I", data[-4:])[0] != zlib.crc32(data[:-4]):
        raise ValueError("T0B CRC checksum mismatch")
    reader = _Reader(data[:-4])
    reader.take(len(MAGIC))
    if reader.byte() != VERSION:
        raise ValueError("unsupported T0B version")
    mode_id = reader.byte()
    if mode_id not in (0, 1):
        raise ValueError("bad float mode")
    fn_id = reader.varint()
    arg_count = reader.varint()
    if not 1 <= arg_count <= 8:
        raise ValueError("invalid arity")
    try:
        args = tuple(REVERSE_TYPE_IDS[reader.byte()] for _ in range(arg_count))
        ret = REVERSE_TYPE_IDS[reader.byte()]
    except KeyError as exc:
        raise ValueError("unknown type ID") from exc
    count = reader.varint()
    if count > 10000:
        raise ValueError("node count limit exceeded")
    argc = {"gt": 2, "mul": 2, "sel": 3, "sum": 1, "rp2": 1,
            "dot": 2, "countp": 1, "max": 1, "mean": 1}
    nodes = []
    for _ in range(count):
        opcode = reader.byte()
        if opcode not in REV_OPCODES:
            raise ValueError("unknown opcode")
        op = REV_OPCODES[opcode]
        nodes.append(Node(op, tuple(reader.ref() for _ in range(argc[op]))))
    result = reader.ref()
    if reader.f.read():
        raise ValueError("unexpected bytes after return")
    p = Program(fn_id, args, ret, "strict" if mode_id == 0 else "fast", tuple(nodes), result)
    validate(p)
    return p
