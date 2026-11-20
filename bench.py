"""Reproducible, *local only* microbenchmark. Use: python bench.py --size 200000"""
from __future__ import annotations
import argparse
import ctypes
from pathlib import Path
import random
import statistics
import tempfile
import time

from t0 import (natural_to_t0, optimize, encode, format_program, compile_native,
                NativeKernel, encode_semantic_tokens)


def median_ms(func, trials: int = 9) -> float:
    times = []
    for _ in range(trials):
        t0 = time.perf_counter_ns()
        func()
        times.append((time.perf_counter_ns() - t0) / 1e6)
    return statistics.median(times)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=200_000)
    parser.add_argument("--trials", type=int, default=9)
    a = parser.parse_args()
    values = [random.Random(73 + i).uniform(-10, 10) for i in range(a.size)]
    p = natural_to_t0("실수 배열에서 양수만 제곱한 후 전부 더해줘")
    optimized = optimize(p)
    with tempfile.TemporaryDirectory() as scratch:
        lib = compile_native(optimized, Path(scratch) / "t0.so")
        kernel = NativeKernel(optimized, lib)
        prepared = kernel.prepare(values)
        native_value = kernel.execute_prepared(prepared)
        # Same algorithm in pure Python; Python float uses binary64,
        # so outputs can differ slightly from strict binary32.
        def py_loop():
            accum = 0.0
            for x in values:
                if x > 0:
                    accum += x * x
            return accum
        py_value = py_loop()
        py_time = median_ms(py_loop, a.trials)
        native_time = median_ms(lambda: kernel.execute_prepared(prepared), a.trials)
        source_bytes = len(format_program(p, compact=True).encode("utf-8"))
        opt_bytes = len(format_program(optimized, compact=True).encode("utf-8"))
        raw_bytes = len(encode(p))
        binary_bytes = len(encode(optimized))
        input_tokens = len(encode_semantic_tokens(p))
        fused_tokens = len(encode_semantic_tokens(optimized))
        print(f"Input: {a.size:,} f32 elements; {a.trials} trials (median)")
        print(f"T0 source: {source_bytes} bytes -> optimized {opt_bytes} bytes")
        print(f"T0 binary: {raw_bytes} bytes -> optimized {binary_bytes} bytes")
        print(f"Custom semantic vocabulary: {input_tokens} tokens -> {fused_tokens} tokens")
        print(f"Python loop: {py_time:.3f} ms | result={py_value:.6f}")
        print(f"T0 native: {native_time:.3f} ms | result={native_value:.6f}")
        print(f"Speed ratio: {py_time / native_time:.1f}x (loop only; compile/transfer excluded)")
        print("Note: semantic IDs are *not* existing LLM tokens and require a retrained/custom vocabulary.")
        print("Note: Python uses binary64; T0 uses binary32; neither timing includes generation/compilation.")


if __name__ == "__main__":
    main()
