"""Readable T0 assembler and intentionally limited natural-language frontend."""
from __future__ import annotations

import re
from .ir import Node, Program, Ref, validate, f32

_HEADER = re.compile(
    r"\s*F(\d+)\(([^)]*)\)->(f32|u64)!p~(strict|fast)\{(.*)\}\s*",
    re.DOTALL,
)
_IMM = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?f")
_FUN = re.compile(r"([a-z][a-z0-9]*)\((.*)\)", re.DOTALL)


def _split_args(s: str) -> list[str]:
    if not s.strip():
        return []
    parts = []
    depth = 0
    start = 0
    for i, char in enumerate(s):
        if char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth < 0:
                raise ValueError("unbalanced parentheses")
        elif char == "," and depth == 0:
            parts.append(s[start:i].strip())
            start = i + 1
    if depth:
        raise ValueError("unbalanced parentheses")
    parts.append(s[start:].strip())
    if any(not x for x in parts):
        raise ValueError("empty operand")
    return parts


def parse_program(source: str) -> Program:
    match = _HEADER.fullmatch(source)
    if match is None:
        raise ValueError("invalid T0 header (expected F0(f32[*])->f32!p~strict{...})")
    fn_id, args, ret, mode, body = match.groups()
    arg_types = tuple(x.strip() for x in args.split(",") if x.strip())
    nodes: list[Node] = []

    def expr(code: str) -> Ref:
        code = code.strip()
        if re.fullmatch(r"@\d+", code):
            return Ref.arg(int(code[1:]))
        if re.fullmatch(r"%\d+", code):
            idx = int(code[1:])
            if idx >= len(nodes):
                raise ValueError(f"forward or undefined reference %{idx}")
            return Ref.tmp(idx)
        if _IMM.fullmatch(code):
            return Ref.imm(float(code[:-1]))
        call = _FUN.fullmatch(code)
        if not call:
            raise ValueError(f"unrecognized expression: {code}")
        op, operands_src = call.groups()
        operands = tuple(expr(part) for part in _split_args(operands_src))
        nodes.append(Node(op, operands))
        return Ref.tmp(len(nodes) - 1)

    result = None
    # Semicolons and newlines delimit statements; they may not appear in literals.
    for statement in re.split(r"[;\n]+", body):
        s = statement.strip()
        if not s:
            continue
        if s.startswith("R="):
            if result is not None:
                raise ValueError("multiple returns")
            result = expr(s[2:])
        else:
            name, sep, rhs = s.partition("=")
            if not sep or not re.fullmatch(r"%\d+", name):
                raise ValueError(f"invalid statement: {s}")
            if result is not None:
                raise ValueError("statement after return")
            expected = len(nodes)
            if int(name[1:]) != expected:
                raise ValueError(f"SSA variables must be assigned in order: expected %{expected}")
            before = len(nodes)
            ref = expr(rhs)
            if len(nodes) != before + 1 or ref != Ref.tmp(before):
                raise ValueError("assignment must have one top-level operation")
    if result is None:
        raise ValueError("missing R= return")
    program = Program(int(fn_id), arg_types, ret, mode, tuple(nodes), result)
    validate(program)
    return program


class UnsupportedNaturalLanguage(ValueError):
    pass


def natural_to_t0(request: str) -> Program:
    """Small *rule-based* Korean/English demo. Not an LLM or general NL compiler.

    Ambiguous/unsupported inputs are rejected rather than silently miscompiled.
    """
    s = re.sub(r"\s+", "", request.lower())
    if not s:
        raise UnsupportedNaturalLanguage("empty request")
    # The pattern matcher must not silently ignore a negation or an alternate
    # condition. We deliberately fail closed for common out-of-scope intent.
    blockers = ("아닌", "않", "말고", "제외", "빼고", "음수", "절댓값",
                "최솟값", "최소값", "역순", "정렬", "not", "except",
                "without", "negative", "exclude", "minimum")
    if any(w in s for w in blockers):
        raise UnsupportedNaturalLanguage("부정/제외/지원하지 않는 조건이 포함되어 안전하게 변환할 수 없습니다")

    has_positive = any(w in s for w in ("양수", "positive", ">0", "0보다큰"))
    has_square = any(w in s for w in ("제곱", "square", "squared", "x*x"))
    has_sum = any(w in s for w in ("합", "더해", "더하", "sum", "total", "모두더", "전부더"))
    has_count = any(w in s for w in ("개수", "몇개", "갯수", "count", "세줘", "세어"))
    is_dot = any(w in s for w in ("내적", "dotproduct", "dot(", "점곱"))
    is_max = any(w in s for w in ("최댓값", "최대값", "가장큰", "maximum", "maxvalue"))
    is_mean = any(w in s for w in ("평균", "average", "mean"))

    if is_dot and not any((has_positive, has_square, has_count, is_max, is_mean)):
        return parse_program("F0(f32[*],f32[*])->f32!p~strict{R=sum(mul(@0,@1))}")
    if has_positive and has_square and has_sum and not any((has_count, is_max, is_mean, is_dot)):
        return parse_program("F0(f32[*])->f32!p~strict{"
                             "%0=gt(@0,0f);%1=mul(@0,@0);"
                             "%2=sel(%0,%1,0f);R=sum(%2)}")
    if has_positive and has_count and not any((has_square, is_max, is_mean, is_dot)):
        return parse_program("F0(f32[*])->u64!p~strict{R=countp(@0)}")
    if is_max and not any((has_positive, has_square, has_count, is_mean, is_dot)):
        return parse_program("F0(f32[*])->f32!p~strict{R=max(@0)}")
    if is_mean and not any((has_positive, has_square, has_count, is_max, is_dot)):
        return parse_program("F0(f32[*])->f32!p~strict{R=mean(@0)}")
    if has_sum and not any((has_positive, has_square, has_count, is_max, is_mean, is_dot)):
        return parse_program("F0(f32[*])->f32!p~strict{R=sum(@0)}")
    raise UnsupportedNaturalLanguage(
        "지원하지 않거나 모호한 자연어입니다. 지원 예: '양수만 제곱해서 합산해', "
        "'두 배열의 내적', '양수 개수', '배열 합계', '최댓값', '평균'."
    )
