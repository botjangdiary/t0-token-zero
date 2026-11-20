"""T0-R correctness, accounting, and selective retry tests."""
import json

import pytest

from t0.reasoning import (Budget, BudgetExceeded, Check, Completion, Engine,
                          Graph, Record, State, Step, Usage, VerificationFailed)
from t0.reasoning_bench import benchmark


def _tiny():
    return Graph([Step("z", "combine results", ("a", "b")),
                  Step("b", "independent branch"), Step("a", "source")])


def test_graph_topological_dependencies():
    g = _tiny()
    assert [x.id for x in g.steps] == ["a", "b", "z"] or [x.id for x in g.steps] == ["b", "a", "z"]
    assert g.ancestors("z") == {"a", "b"}
    assert g.ancestors("a") == set()


@pytest.mark.parametrize("steps", [
    [Step("a", "A"), Step("a", "B")],
    [Step("a", "A", ("missing",))],
    [Step("a", "A", ("b",)), Step("b", "B", ("a",))],
])
def test_invalid_graphs(steps):
    with pytest.raises(ValueError):
        Graph(steps)


def test_verified_dependency_context_and_resume():
    g = Graph([Step("spec", "spec"), Step("unrelated", "unrelated"),
               Step("test", "test", ("spec",))])
    def model(step, prompt, cap):
        return Completion(step.id + "-accepted")
    def validator(step, result, deps):
        return Check(result == step.id + "-accepted")
    state, report = Engine(g, model, validator, goal="G").run()
    assert report.calls == 3
    assert "unrelated-accepted" not in report.prompts[-1]
    assert "spec-accepted" in report.prompts[-1]
    resumed, repeat = Engine(g, model, validator, goal="G").run(state)
    assert resumed is state
    assert repeat.calls == 0
    assert repeat.reused == 3
    assert State.decode(state.encode()) == state


def test_cache_invalidation_maintains_unrelated_results():
    g = Graph([Step("root", "one"), Step("leaf", "two", ("root",)), Step("other", "three")])
    calls = []
    def model(step, prompt, cap):
        calls.append(step.id)
        return Completion("reply:" + step.instruction + ":" + prompt[:5])
    def validator(step, result, deps):
        return Check(True)
    state, _ = Engine(g, model, validator, goal="G").run()
    # Change task instruction, but preserve unrelated cached artifacts.
    changed = Graph([Step("root", "CHANGED"), Step("leaf", "two", ("root",)), Step("other", "three")])
    _, second = Engine(changed, model, validator, goal="G").run(state)
    assert second.executed == ["root", "leaf"]
    assert second.reused == 1


def test_selective_retry_bounded_feedback():
    g = Graph([Step("a", "provide correct artifact")])
    seen = []
    def model(step, prompt, cap):
        seen.append(json.loads(prompt))
        return Completion("not yet" if len(seen) == 1 else "valid")
    def validator(step, result, deps):
        return Check(result == "valid", "try again" + "!" * 800)
    state, report = Engine(g, model, validator, goal="G", max_retries=1).run()
    assert report.calls == 2 and report.failed_attempts == 1
    assert state.records["a"].attempts == 2
    assert seen[1]["last_verifier_feedback"].startswith("try again")
    assert len(seen[1]["last_verifier_feedback"]) == 500
    assert "not yet" not in json.dumps(seen[1])


def test_exhausted_retry_rejects_invalid_artifact():
    g = Graph([Step("a", "instruction")])
    state = State()
    with pytest.raises(VerificationFailed):
        Engine(g, lambda *args: Completion("bad"), lambda *args: Check(False, "wrong"),
               goal="G", max_retries=1).run(state)
    assert state.records == {}


def test_fail_closed_for_prompt_and_total_budget():
    g = Graph([Step("a", "required")])
    model_calls = []
    def model(*args):
        model_calls.append(1)
        return Completion("ok")
    for budget in (Budget(1, 10, 100), Budget(1000, 100, 10)):
        with pytest.raises(BudgetExceeded):
            Engine(g, model, lambda *args: Check(True), goal="G", budget=budget).run()
    assert model_calls == []


def test_output_cap_and_provider_usage():
    g = Graph([Step("a", "required")])
    with pytest.raises(BudgetExceeded):
        Engine(g, lambda *args: Completion("long"), lambda *args: Check(True),
               goal="G", budget=Budget(1000, 1, 2000)).run()
    _, r = Engine(g, lambda *args: Completion("ok", Usage(3, 2)),
                  lambda *args: Check(True), goal="G").run()
    assert r.provider_reported_tokens
    assert (r.input_units, r.output_units) == (3, 2)
    assert r.provider_reasoning_tokens is None
    _, r2 = Engine(g, lambda *args: Completion("ok", Usage(8, 5, 3)),
                   lambda *args: Check(True), goal="G").run()
    assert r2.provider_reasoning_tokens == 3


def test_state_integrity_rejection():
    state = State({"a": Record("data", "a"*64, "0"*64, 1)})
    with pytest.raises(ValueError):
        State.decode(state.encode())
    for bad in ("{}", "[]", "null", "{\"version\":2}"):
        with pytest.raises(ValueError):
            State.decode(bad)


def test_no_model_benchmark_is_reproducible_and_verified():
    a, b = benchmark(), benchmark()
    assert a == b
    assert a["baseline_full_context"]["verified_steps"] == 9
    assert a["t0r_dependency_scoped"]["verified_steps"] == 9
    assert a["input_context_reduction_pct"] > 0
    assert a["retry_simulation"]["additional_calls_above_one_per_step"] == 1
    assert a["resume_simulation"]["calls_for_same_verified_artifact"] == 1


def test_cli_reason_bench(capsys):
    from t0.__main__ import main
    assert main(["reason-bench", "--json"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["input_context_reduction_pct"] > 0
