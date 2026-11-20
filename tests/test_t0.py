import math
import random
import struct
import subprocess
import sys

import pytest

from t0 import (natural_to_t0, parse_program, optimize, encode, decode,
                run_native, run_reference, format_program, generate_c)
from t0.ir import Ref, Node, Program


def test_natural_variants():
    cases = [
        ("실수 배열에서 양수만 제곱한 후 전부 더해줘", "rp2"),
        ("두 배열의 내적을 구해줘", "dot"),
        ("배열에서 양수의 개수를 세줘", "countp"),
        ("배열의 합계를 구해줘", "sum"),
        ("배열 최댓값", "max"),
        ("배열의 평균", "mean"),
        ("positive values squared then sum", "rp2"),
    ]
    for request, op in cases:
        assert optimize(natural_to_t0(request)).nodes[-1].op == op


def test_unknown_request_is_rejected():
    with pytest.raises(ValueError):
        natural_to_t0("웹사이트 만들어줘")
    with pytest.raises(ValueError):
        natural_to_t0("음수만 제곱해줘")
    with pytest.raises(ValueError):
        natural_to_t0("양수는 제곱하지 말고 합산해")


@pytest.mark.parametrize("source", [
    "F0(f32[*])->f32!p~strict{R=sum(@0)}",
    "F0(f32[*])->f32!p~fast{R=rp2(@0)}",
    "F2(f32[*],f32[*])->f32!p~strict{R=sum(mul(@0,@1))}",
    "F0(f32[*])->u64!p~strict{R=countp(@0)}",
    "F0(f32[*])->f32!p~strict{R=sum(sel(gt(@0,0f),mul(@0,@0),0f))}",
    "F0(f32[*])->f32!p~strict{R=max(@0)}",
    "F0(f32[*])->f32!p~strict{R=mean(@0)}",
])
def test_roundtrip_text_binary_text(source):
    a = parse_program(source)
    b = decode(encode(a))
    assert a == b
    assert parse_program(format_program(a)) == a


def test_binary_corruption():
    blob = bytearray(encode(natural_to_t0("배열 합계")))
    blob[5] ^= 1
    with pytest.raises(ValueError, match="checksum"):
        decode(bytes(blob))
    with pytest.raises(ValueError):
        decode(bytes(blob[:-1]))


def test_type_and_reference_checker():
    for text in [
        "F0(f32[*])->f32!p~strict{R=sum(%3)}",  # forward ref
        "F0(f32[*])->f32!p~strict{R=mul(@0,@0)}",  # scalar return of vector
        "F0(f32[*])->f32!p~strict{R=sum(gt(@0,0f))}",  # sum of bool
        "F0(f32[*])->f32!p~strict{R=dot(@0,1f)}",  # dot type
        "F0(f32[*])->f32!p~strict{%0=sum(@0);%2=sum(@0);R=%0}",
        "F0(f32[*])->u64!p~strict{R=sum(@0)}",
    ]:
        with pytest.raises(ValueError):
            parse_program(text)


def test_rewrites_only_if_valid():
    p = natural_to_t0("양수만 제곱해서 합해줘")
    q = optimize(p)
    assert q.nodes == (Node("rp2", (Ref.arg(0),)),)
    assert len(encode(q)) < len(encode(p))
    # Squaring something else must NOT be rewritten to rp2.
    invalid = parse_program("F0(f32[*],f32[*])->f32!p~strict{R=sum(sel(gt(@0,0f),mul(@0,@1),0f))}")
    assert optimize(invalid) == invalid
    same = parse_program("F0(f32[*])->f32!p~strict{R=sum(mul(@0,@0))}")
    # dot requires distinct args in this implementation.
    assert optimize(same) == same


@pytest.mark.parametrize("source,arrays,expected", [
    ("F0(f32[*])->f32!p~strict{R=rp2(@0)}", [[-4, 1, 3]], 10),
    ("F0(f32[*])->f32!p~strict{R=sum(@0)}", [[-4, 1, 3]], 0),
    ("F0(f32[*])->u64!p~strict{R=countp(@0)}", [[-4, 0, 1, 3]], 2),
    ("F0(f32[*],f32[*])->f32!p~strict{R=dot(@0,@1)}", [[1, 2, 3], [10, -2, 1]], 9),
    ("F0(f32[*])->f32!p~strict{R=max(@0)}", [[-4, 1, 3]], 3),
    ("F0(f32[*])->f32!p~strict{R=mean(@0)}", [[-4, 1, 3]], 0),
    ("F0(f32[*])->f32!p~strict{R=mean(@0)}", [[1, 2, 3]], 2),
    ("F0(f32[*])->f32!p~strict{R=sum(sel(gt(@0,0f),mul(@0,@0),0f))}", [[-4, 1, 3]], 10),
])
def test_reference_and_native(source, arrays, expected):
    p = parse_program(source)
    interpreted = run_reference(p, *arrays)
    native = run_native(p, *arrays)
    assert interpreted == pytest.approx(expected, rel=1e-5)
    assert native == pytest.approx(expected, rel=1e-5)
    assert native == pytest.approx(interpreted, rel=1e-5)


def test_nested_reductions_and_elementwise_fusion():
    # A scalar reduction used inside a second vector expression.
    p = parse_program("F0(f32[*])->f32!p~strict{%0=sum(@0);R=sum(mul(@0,%0))}")
    assert run_native(p, [1, 2, 3]) == pytest.approx(36)
    assert run_reference(p, [1, 2, 3]) == pytest.approx(36)
    # condition broadcasting
    q = parse_program("F0(f32[*])->f32!p~strict{%0=gt(2f,1f);R=sum(sel(%0,@0,0f))}")
    assert run_native(q, [1, 2, 3]) == pytest.approx(6)


def test_zero_length():
    assert run_native(natural_to_t0("배열 합계"), []) == 0.0
    assert run_native(natural_to_t0("양수 개수"), []) == 0
    for phrase in ("배열 평균", "배열 최댓값"):
        with pytest.raises(ValueError, match="nonempty"):
            run_native(natural_to_t0(phrase), [])


def test_mismatched_array_lengths():
    p = natural_to_t0("두 배열의 내적")
    with pytest.raises(ValueError, match="same length"):
        run_native(p, [1, 2], [1])


def test_random_regression():
    rng = random.Random(1234)
    cases = [natural_to_t0(v) for v in (
        "양수만 제곱해서 합해줘", "두 배열의 내적", "양수 개수", "배열 합계", "배열 최댓값", "평균"
    )]
    a = [rng.uniform(-10, 10) for _ in range(1000)]
    b = [rng.uniform(-10, 10) for _ in range(1000)]
    for p in cases:
        arrays = [a, b] if len(p.arg_types) == 2 else [a]
        # Float reductions are deliberately ordered, not algebraically reordered.
        assert run_native(p, *arrays) == pytest.approx(run_reference(p, *arrays), abs=1e-3, rel=1e-5)


def test_compiled_c_contains_no_temporary_arrays():
    p = natural_to_t0("실수 배열에서 양수만 제곱한 후 전부 더해줘")
    code = generate_c(p)
    assert "for (size_t i = 0; i < n; ++i)" in code
    assert "malloc" not in code
    assert "if (x > 0.0f)" in code


def test_dedicated_token_vocab_roundtrip():
    from t0 import encode_semantic_tokens, decode_semantic_tokens
    cases = [
        natural_to_t0("양수만 제곱해서 합해줘"),
        natural_to_t0("두 배열의 내적"),
        natural_to_t0("배열의 평균"),
        parse_program("F2(f32[*])->f32!p~fast{R=sum(mul(@0,0.75f))}"),
        parse_program("F0(f32[*])->f32!p~strict{R=0.5f}"),
        parse_program("F0(f32[*])->f32!p~strict{R=0f}"),
        parse_program("F0(f32[*])->f32!p~strict{R=-1f}"),
    ]
    for p in cases:
        ids = encode_semantic_tokens(p)
        assert decode_semantic_tokens(ids) == p
    p = optimize(cases[0])
    assert encode_semantic_tokens(p) == [16, 69, 128, 3, 1]
    assert len(encode_semantic_tokens(p)) < len(encode_semantic_tokens(cases[0]))


def test_token_stream_invalid():
    from t0 import decode_semantic_tokens
    for seq in ([16, 69, 128, 3], [16, 69, 128, 3, 1, 1],
                [16, 69, 129, 3, 1], [16, 69, 128, 4, 160, 0]):
        with pytest.raises(ValueError):
            decode_semantic_tokens(seq)


def test_one_command_ask(capsys):
    from t0.__main__ import main
    code = main(["ask", "양수만 제곱해서 합산해줘", "--data", "1,-2,3",
                 "--show-ir", "--show-tokens"])
    assert code == 0
    out = capsys.readouterr().out
    assert "%0=rp2(@0)" in out
    assert "[16, 69, 128, 3, 1]" in out
    assert "Result: 10.0" in out
