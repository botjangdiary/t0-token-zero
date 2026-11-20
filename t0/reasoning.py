"""T0-R v0.2: verified dependency-scoped reasoning orchestration.

This module reduces *supplied context* and redundant model calls. It does not
intercept, read, or compress a provider's private chain-of-thought tokens.
No model API is included; callers inject a completion function and token meter.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
import re
from typing import Callable, Literal, Mapping

_ID = re.compile(r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha(value: object) -> str:
    return sha256(_canonical(value).encode("utf-8")).hexdigest()


def utf8_bytes(text: str) -> int:
    """Deterministic character-size proxy; NOT a tokenizer."""
    return len(text.encode("utf-8"))


@dataclass(frozen=True)
class Step:
    id: str
    instruction: str
    needs: tuple[str, ...] = ()
    contract: str = ""

    def __post_init__(self) -> None:
        if not _ID.fullmatch(self.id):
            raise ValueError(f"invalid step identifier: {self.id!r}")
        if not self.instruction.strip():
            raise ValueError(f"step {self.id}: empty instruction")
        if len(set(self.needs)) != len(self.needs):
            raise ValueError(f"step {self.id}: duplicate dependency")


@dataclass(frozen=True)
class Check:
    ok: bool
    feedback: str = ""


@dataclass(frozen=True)
class Usage:
    """Actual provider-reported token usage, when available."""
    input_tokens: int
    output_tokens: int
    reasoning_tokens: int | None = None  # Provider-specific subset of output tokens.

    def __post_init__(self) -> None:
        if self.input_tokens < 0 or self.output_tokens < 0:
            raise ValueError("usage values must be nonnegative")
        if self.reasoning_tokens is not None and (self.reasoning_tokens < 0 or self.reasoning_tokens > self.output_tokens):
            raise ValueError("invalid reasoning token usage")


@dataclass(frozen=True)
class Completion:
    text: str
    usage: Usage | None = None


@dataclass(frozen=True)
class Record:
    value: str
    signature: str
    digest: str
    attempts: int


@dataclass
class State:
    """Verified, reusable artifacts only. No private chain-of-thought logs."""
    records: dict[str, Record] = field(default_factory=dict)

    def encode(self) -> str:
        data = {"version": 1, "records": {
            k: {"value": v.value, "signature": v.signature,
                "digest": v.digest, "attempts": v.attempts}
            for k, v in sorted(self.records.items())}}
        return _canonical(data)

    @classmethod
    def decode(cls, raw: str) -> State:
        data = json.loads(raw)
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("unsupported T0-R state format")
        raw_records = data.get("records")
        if not isinstance(raw_records, dict):
            raise ValueError("invalid records")
        records: dict[str, Record] = {}
        for key, r in raw_records.items():
            if not isinstance(key, str) or not _ID.fullmatch(key) or not isinstance(r, dict):
                raise ValueError("invalid record")
            try:
                rec = Record(r["value"], r["signature"], r["digest"], r["attempts"])
            except (KeyError, TypeError) as e:
                raise ValueError("invalid record fields") from e
            if (not isinstance(rec.value, str) or not isinstance(rec.signature, str)
                    or len(rec.signature) != 64 or not isinstance(rec.digest, str)
                    or rec.digest != sha256(rec.value.encode("utf-8")).hexdigest()
                    or type(rec.attempts) is not int or rec.attempts < 1):
                raise ValueError("invalid record integrity")
            records[key] = rec
        return cls(records)


@dataclass(frozen=True)
class Budget:
    max_prompt_units: int = 16_000
    max_output_units: int = 4_000
    max_total_units: int = 200_000

    def __post_init__(self) -> None:
        if min(self.max_prompt_units, self.max_output_units, self.max_total_units) < 1:
            raise ValueError("budgets must be positive")


class BudgetExceeded(RuntimeError):
    pass


class VerificationFailed(RuntimeError):
    pass


@dataclass
class RunReport:
    calls: int = 0
    reused: int = 0
    failed_attempts: int = 0
    input_units: int = 0
    output_units: int = 0
    provider_reported_tokens: bool = True
    provider_reasoning_tokens: int | None = 0
    prompts: list[str] = field(default_factory=list)
    executed: list[str] = field(default_factory=list)

    @property
    def total_units(self) -> int:
        return self.input_units + self.output_units


class Graph:
    """Validated, acyclic dependency graph with stable execution order."""

    def __init__(self, steps: list[Step] | tuple[Step, ...]):
        if not steps:
            raise ValueError("graph must have at least one step")
        by_id: dict[str, Step] = {}
        for s in steps:
            if s.id in by_id:
                raise ValueError(f"duplicate step: {s.id}")
            by_id[s.id] = s
        for s in steps:
            if any(n not in by_id for n in s.needs):
                raise ValueError(f"step {s.id}: unknown dependency")
            if s.id in s.needs:
                raise ValueError(f"step {s.id}: self dependency")
        visiting: set[str] = set()
        visited: set[str] = set()
        ordered: list[Step] = []

        def visit(id_: str) -> None:
            if id_ in visiting:
                raise ValueError("dependency cycle detected")
            if id_ in visited:
                return
            visiting.add(id_)
            for parent in by_id[id_].needs:
                visit(parent)
            visiting.remove(id_)
            visited.add(id_)
            ordered.append(by_id[id_])

        for s in steps:
            visit(s.id)
        self.steps = tuple(ordered)
        self.by_id = by_id

    def ancestors(self, id_: str) -> set[str]:
        parents: set[str] = set()

        def walk(cur: str) -> None:
            for parent in self.by_id[cur].needs:
                if parent not in parents:
                    parents.add(parent)
                    walk(parent)
        walk(id_)
        return parents


# The injected model must return a completion, not a hidden reasoning trace.
Model = Callable[[Step, str, int], Completion]
Validator = Callable[[Step, str, Mapping[str, str]], Check]
Meter = Callable[[str], int]


class Engine:
    """Run a verified graph with bounded feedback and incremental reuse.

    `meter` should be the actual model tokenizer for genuine prompt token caps.
    Defaults to UTF-8 bytes, a clearly labeled deterministic size proxy.
    If provider Usage is supplied, reports use those actual billed token counts.
    """

    def __init__(self, graph: Graph, model: Model, validator: Validator, *,
                 goal: str, mode: Literal["scoped", "full"] = "scoped",
                 budget: Budget | None = None, meter: Meter = utf8_bytes,
                 max_retries: int = 1, verification_version: str = "v1"):
        if not goal.strip():
            raise ValueError("goal must not be empty")
        if mode not in ("scoped", "full"):
            raise ValueError("mode must be scoped or full")
        if max_retries < 0:
            raise ValueError("negative retry limit")
        self.graph = graph
        self.model = model
        self.validator = validator
        self.goal = goal
        self.mode = mode
        self.budget = budget or Budget()
        self.meter = meter
        self.max_retries = max_retries
        self.verification_version = verification_version

    def _signature(self, step: Step, deps: Mapping[str, str]) -> str:
        return _sha({"goal": self.goal, "task": step.instruction,
                     "contract": step.contract, "needs": list(step.needs),
                     "deps": deps, "verification_version": self.verification_version})

    def _prompt(self, step: Step, state: State, feedback: str) -> str:
        allowed = (self.graph.ancestors(step.id) if self.mode == "scoped"
                   else set(state.records))
        # The prompt carries only verified records, never unbounded model chatter.
        context = {s.id: state.records[s.id].value for s in self.graph.steps
                   if s.id in allowed and s.id in state.records}
        return _canonical({"goal": self.goal, "step": step.id,
                           "task": step.instruction, "contract": step.contract,
                           "verified_context": context, "last_verifier_feedback": feedback})

    def run(self, state: State | None = None) -> tuple[State, RunReport]:
        state = state if state is not None else State()
        report = RunReport()
        for step in self.graph.steps:
            deps = {name: state.records[name].value for name in step.needs}
            signature = self._signature(step, deps)
            previous = state.records.get(step.id)
            if previous and previous.signature == signature:
                # Cached state was checked and contains a content checksum.
                if previous.digest == sha256(previous.value.encode("utf-8")).hexdigest():
                    report.reused += 1
                    continue
            # Old output must not enter full-history prompts while recomputing.
            state.records.pop(step.id, None)
            feedback = ""
            for attempt in range(self.max_retries + 1):
                prompt = self._prompt(step, state, feedback)
                prompt_units = self.meter(prompt)
                if prompt_units < 0:
                    raise ValueError("meter returned negative count")
                if prompt_units > self.budget.max_prompt_units:
                    raise BudgetExceeded(f"step {step.id}: prompt over budget")
                # Reserve the entire configured response cap, before calling model.
                if report.total_units + prompt_units + self.budget.max_output_units > self.budget.max_total_units:
                    raise BudgetExceeded(f"step {step.id}: insufficient total budget")
                completion = self.model(step, prompt, self.budget.max_output_units)
                if not isinstance(completion, Completion) or not isinstance(completion.text, str):
                    raise TypeError("model must return Completion(text, usage?)")
                measured_output = self.meter(completion.text)
                if measured_output > self.budget.max_output_units:
                    raise BudgetExceeded(f"step {step.id}: model exceeded output cap")
                if completion.usage is None:
                    report.provider_reported_tokens = False
                    report.provider_reasoning_tokens = None
                    input_used, output_used = prompt_units, measured_output
                else:
                    input_used, output_used = (completion.usage.input_tokens,
                                               completion.usage.output_tokens)
                    if completion.usage.reasoning_tokens is None:
                        report.provider_reasoning_tokens = None
                    elif report.provider_reasoning_tokens is not None:
                        report.provider_reasoning_tokens += completion.usage.reasoning_tokens
                    if output_used > self.budget.max_output_units:
                        raise BudgetExceeded(f"step {step.id}: actual output tokens over cap")
                report.input_units += input_used
                report.output_units += output_used
                report.calls += 1
                report.prompts.append(prompt)
                report.executed.append(step.id)
                if report.total_units > self.budget.max_total_units:
                    raise BudgetExceeded("provider usage exceeded total budget")
                check = self.validator(step, completion.text, deps)
                if not isinstance(check, Check):
                    raise TypeError("validator must return Check(ok, feedback)")
                if check.ok:
                    state.records[step.id] = Record(
                        value=completion.text,
                        signature=signature,
                        digest=sha256(completion.text.encode("utf-8")).hexdigest(),
                        attempts=attempt + 1)
                    break
                report.failed_attempts += 1
                # Keep feedback compact. It is diagnostic input, NOT authority.
                feedback = check.feedback[:500]
                if attempt == self.max_retries:
                    raise VerificationFailed(f"step {step.id} failed verification: {feedback}")
        return state, report
