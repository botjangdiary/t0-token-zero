"""Reproducible no-LLM benchmark for T0-R context routing.

All byte counts measure *serialized prompts*, NOT actual model tokens.
The 'oracle' deliberately returns fixed expected artifacts; this benchmark
measures orchestration overhead and invariants, not AI reasoning quality.
"""
from __future__ import annotations

import argparse
import json
from typing import Mapping

from .reasoning import Check, Completion, Engine, Graph, State, Step, utf8_bytes


GOAL = "Build an auditable command-line data processing tool with correct outputs."

# Independent branches are intentional: full-chat history is redundant for
# unrelated nodes, while verified dependency artifacts remain available.
STEPS = (
    Step("spec", "Describe the accepted input schema and observable behavior", contract="one checked specification"),
    Step("parser", "Implement strict input parsing and reject invalid records", ("spec",)),
    Step("math", "Derive a numerically stable aggregation strategy", ("spec",)),
    Step("security", "List input limits and security invariants", ("spec",)),
    Step("tests", "Build regression test fixtures from the specification", ("spec",)),
    Step("doc", "Write a short command reference", ("spec",)),
    Step("integration", "Integrate parser and math with tests", ("parser", "math", "tests")),
    Step("audit", "Review implementation against security requirements", ("integration", "security")),
    Step("release", "Produce a verified release manifest", ("audit", "doc")),
)

ARTIFACTS = {
    "spec": "Schema: each row has a finite decimal value; reject nulls and nonnumeric fields; output count and sum. ",
    "parser": "Parse UTF-8 rows; bound row length; validate decimal values; reject unexpected fields. ",
    "math": "Use ordered float64 accumulation with an explicit precision tolerance; do not reorder sums. ",
    "security": "Limit total input bytes; reject malformed UTF-8; avoid file writes beyond output path. ",
    "tests": "Fixtures: one positive, one zero, one negative; reject invalid and overflow-like strings. ",
    "doc": "Usage: tool --input rows.csv --output report.json; exits nonzero for malformed data. ",
    "integration": "The parser yields validated finite values; aggregator emits count and sum; tests pass. ",
    "audit": "Limits enforced and malformed input rejected before aggregation; reviewed security invariants. ",
    "release": "Manifest: validated spec, parser, aggregation, tests, documentation and audit included. ",
}
# Retain a meaningful payload to approximate longer developer artifacts,
# without fabricating an LLM's chain of thought.
OUTPUTS = {key: value * 4 for key, value in ARTIFACTS.items()}


def oracle(step: Step, prompt: str, max_output: int) -> Completion:
    # Deterministic 'fake model'. Results do not depend on the prompt.
    return Completion(OUTPUTS[step.id])


def verify(step: Step, text: str, dependencies: Mapping[str, str]) -> Check:
    return Check(text == OUTPUTS[step.id], "artifact mismatch")


def _measure(mode: str) -> tuple[State, dict]:
    graph = Graph(STEPS)
    engine = Engine(graph, oracle, verify, goal=GOAL, mode=mode)
    state, report = engine.run()
    return state, {"model_calls": report.calls,
                   "verified_steps": len(state.records),
                   "prompt_bytes": report.input_units,
                   "response_bytes": report.output_units,
                   "measured_provider_tokens": report.provider_reported_tokens}


def benchmark() -> dict:
    full_state, full = _measure("full")
    scoped_state, scoped = _measure("scoped")
    if {k: v.value for k, v in full_state.records.items()} != {k: v.value for k, v in scoped_state.records.items()}:
        raise AssertionError("strategies did not produce the same verified outputs")

    graph = Graph(STEPS)
    calls_by_step: dict[str, int] = {}

    def faulty_once(step: Step, prompt: str, max_output: int) -> Completion:
        calls_by_step[step.id] = calls_by_step.get(step.id, 0) + 1
        if step.id == "math" and calls_by_step[step.id] == 1:
            return Completion("not a valid aggregation algorithm")
        return oracle(step, prompt, max_output)

    fixed_state, retry_report = Engine(graph, faulty_once, verify, goal=GOAL).run()
    # Invalidating one verified artifact should re-execute only that node and
    # descendants; independent branches keep their verified results.
    fixed_state.records.pop("math")
    refreshed, update_report = Engine(graph, oracle, verify, goal=GOAL).run(fixed_state)
    if len(refreshed.records) != len(STEPS):
        raise AssertionError("unverified final state")
    # The cache keeps descendants if their direct inputs were unchanged; this is
    # intentional. Here 'math' produces the same verified output, so everything
    # downstream is still valid and can be reused.
    saving_pct = (1 - scoped["prompt_bytes"] / full["prompt_bytes"]) * 100
    return {
        "measurement": "UTF-8 prompt bytes (proxy, NOT actual LLM tokens or private reasoning tokens)",
        "workflow_steps": len(STEPS),
        "baseline_full_context": full,
        "t0r_dependency_scoped": scoped,
        "input_context_reduction_pct": round(saving_pct, 2),
        "retry_simulation": {
            "deliberate_failed_artifacts": retry_report.failed_attempts,
            "model_calls": retry_report.calls,
            "additional_calls_above_one_per_step": retry_report.calls - len(STEPS),
            "all_verified": len(fixed_state.records) == len(STEPS),
        },
        "resume_simulation": {
            "invalidated_step": "math",
            "calls_for_same_verified_artifact": update_report.calls,
            "verified_steps_reused": update_report.reused,
        },
        "limitations": "No LLM is called; a deterministic oracle returns fixed outputs. Metrics do not establish accuracy or actual model token savings.",
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T0-R deterministic context routing benchmark")
    parser.add_argument("--json", action="store_true", help="print machine-readable output")
    args = parser.parse_args(argv)
    result = benchmark()
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        base = result["baseline_full_context"]
        scoped = result["t0r_dependency_scoped"]
        print("T0-R v0.2 — deterministic simulator (NOT actual LLM tokens)")
        print(f"Verified steps: {result['workflow_steps']}")
        print(f"Full-history prompt bytes: {base['prompt_bytes']}")
        print(f"Scoped-context prompt bytes: {scoped['prompt_bytes']}")
        print(f"Prompt byte reduction: {result['input_context_reduction_pct']:.2f}%")
        retry = result["retry_simulation"]
        print(f"Verifier rejection: {retry['deliberate_failed_artifacts']}; extra calls: {retry['additional_calls_above_one_per_step']}")
        resume = result["resume_simulation"]
        print(f"Resume: {resume['calls_for_same_verified_artifact']} reruns, {resume['verified_steps_reused']} reused")
        print("No LLM called; output does not imply actual thinking-token savings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
