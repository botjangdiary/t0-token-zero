"""T0 v0.1 — experimental AI-oriented bytecode language."""
from .frontend import natural_to_t0, parse_program
from .ir import format_program, validate
from .optimizer import optimize
from .binary import encode, decode
from .backend import compile_native, generate_c, run_native, NativeKernel
from .reference import run_reference
from .semantic_tokens import encode_semantic_tokens, decode_semantic_tokens

__all__ = ["natural_to_t0", "parse_program", "format_program", "validate", "optimize",
           "encode", "decode", "compile_native", "generate_c", "run_native",
           "NativeKernel", "run_reference", "encode_semantic_tokens", "decode_semantic_tokens"]
