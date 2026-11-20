"""python -m t0 command line interface."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys

from .backend import compile_native, run_native
from .binary import decode, encode
from .frontend import natural_to_t0, parse_program
from .ir import format_program
from .optimizer import optimize
from .reference import run_reference
from .semantic_tokens import encode_semantic_tokens, decode_semantic_tokens


def _load(path: str):
    contents = Path(path).read_bytes()
    return decode(contents) if contents[:4] == b"T0B\x00" else parse_program(contents.decode("utf-8"))


def _array(src: str) -> list[float]:
    if src.startswith("["):
        values = json.loads(src)
        if not isinstance(values, list):
            raise ValueError("expected a JSON numeric array")
        return [float(v) for v in values]
    if not src.strip():
        return []
    return [float(x) for x in src.split(",")]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="t0", description="T0 v0.1 — minimal typed SSA + native compilation")
    subs = parser.add_subparsers(dest="command", required=True)

    nl = subs.add_parser("nl", help="convert a supported Korean/English phrase to T0")
    nl.add_argument("text")
    nl.add_argument("-o", "--output", help="write .t0 source")
    nl.add_argument("--pack", help="write optimized T0B binary")
    nl.add_argument("--optimized", action="store_true", help="show optimized T0 text")

    ask = subs.add_parser("ask", help="natural-language request -> optimized T0 -> execute")
    ask.add_argument("text")
    ask.add_argument("--data", required=True, help="comma-separated or JSON input array")
    ask.add_argument("--other", action="append", default=[])
    ask.add_argument("--backend", choices=("native", "reference"), default="native")
    ask.add_argument("--show-ir", action="store_true")
    ask.add_argument("--show-tokens", action="store_true")

    dump = subs.add_parser("dump", help="disassemble T0 source or binary")
    dump.add_argument("source")
    dump.add_argument("--compact", action="store_true")

    pack = subs.add_parser("pack", help="typecheck and serialize optimized binary")
    pack.add_argument("source")
    pack.add_argument("-o", "--output", required=True)
    pack.add_argument("--no-opt", action="store_true")

    tokens = subs.add_parser("tokens", help="print dedicated LLM semantic token IDs (not existing GPT tokens)")
    tokens.add_argument("source")
    tokens.add_argument("--no-opt", action="store_true")

    detok = subs.add_parser("detokenize", help="parse a JSON list of semantic token IDs to a verified T0B")
    detok.add_argument("tokens", help="JSON list, e.g. '[16,69,128,3,1]'")
    detok.add_argument("-o", "--output", required=True)

    comp = subs.add_parser("compile", help="emit native shared library")
    comp.add_argument("source")
    comp.add_argument("-o", "--output", required=True)
    comp.add_argument("--emit-c", help="also write human-readable generated C")
    comp.add_argument("--no-opt", action="store_true")

    run = subs.add_parser("run", help="evaluate T0 with reference interpreter or native code")
    run.add_argument("source")
    run.add_argument("--data", required=True, help="comma-separated or JSON input array")
    run.add_argument("--other", action="append", default=[], help="additional array for dot etc")
    run.add_argument("--backend", choices=("native", "reference"), default="native")

    args = parser.parse_args(argv)
    try:
        if args.command == "nl":
            raw = natural_to_t0(args.text)
            display = optimize(raw) if args.optimized else raw
            src = format_program(display)
            print(src)
            if args.output:
                Path(args.output).write_text(src + "\n", encoding="utf-8")
                print(f"T0 source written: {args.output}", file=sys.stderr)
            if args.pack:
                payload = encode(optimize(raw))
                Path(args.pack).write_bytes(payload)
                print(f"Optimized binary: {args.pack} ({len(payload)} bytes)", file=sys.stderr)
        elif args.command == "ask":
            p = optimize(natural_to_t0(args.text))
            if args.show_ir:
                print(format_program(p))
            if args.show_tokens:
                print("Semantic tokens:", json.dumps(encode_semantic_tokens(p)))
            arrays = [_array(args.data)] + [_array(x) for x in args.other]
            result = (run_native(p, *arrays) if args.backend == "native"
                      else run_reference(p, *arrays))
            print("Result:", result)
        elif args.command == "dump":
            print(format_program(_load(args.source), compact=args.compact))
        elif args.command == "pack":
            p = _load(args.source)
            payload = encode(p if args.no_opt else optimize(p))
            Path(args.output).write_bytes(payload)
            print(f"{args.output}: {len(payload)} bytes")
        elif args.command == "tokens":
            p = _load(args.source)
            ids = encode_semantic_tokens(p if args.no_opt else optimize(p))
            print(json.dumps(ids))
            print(f"{len(ids)} semantic tokens (requires a custom model vocabulary)", file=sys.stderr)
        elif args.command == "detokenize":
            ids = json.loads(args.tokens)
            if not isinstance(ids, list):
                raise ValueError("expected JSON list of token IDs")
            p = decode_semantic_tokens(ids)
            blob = encode(p)
            Path(args.output).write_bytes(blob)
            print(f"{args.output}: {len(blob)} bytes")
        elif args.command == "compile":
            p = _load(args.source)
            output = compile_native(p, args.output, emit_c=args.emit_c,
                                    optimize_first=not args.no_opt)
            print(f"Native library: {output}")
        elif args.command == "run":
            p = _load(args.source)
            arrays = [_array(args.data)] + [_array(x) for x in args.other]
            result = (run_native(p, *arrays) if args.backend == "native"
                      else run_reference(p, *arrays))
            print(result)
        return 0
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"t0: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
