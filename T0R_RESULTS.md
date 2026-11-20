# T0-R v0.2 Benchmark — reproducible, no model API

## Goal and hypothesis

Hypothesis: forwarding only verified dependency artifacts will reduce serialized
prompt context relative to forwarding all previously verified artifacts, while
preserving information needed by a correctly specified dependency graph.

## Method

Run: `python -m t0 reason-bench --json` from the repository root.

- Nine tasks form a fixed directed acyclic graph, including several parallel
  branches and three later joins.
- Both modes call the **same deterministic oracle** once per task. This oracle
  returns a predeclared correct artifact independent of the prompt.
- The validator compares each artifact with an exact reference value. Thus
  both arms have the same verified outputs **by construction**.
- Baseline `full` prompts include all earlier verified outputs.
- `scoped` prompts include only verified transitive dependencies.
- Input size is counted with `len(prompt.encode("utf-8"))`. The benchmark does
  not use any vendor tokenizer, infer hidden reasoning length, call an LLM, or
  test model accuracy on real tasks.
- Output sizes are identical by construction. No GPU-time measurement is made.

## Results (deterministic)

| Metric | full | scoped |
| --- | ---: | ---: |
| calls | 9 | 9 |
| verified artifacts | 9 | 9 |
| prompt bytes | 15,627 | 10,898 |
| output bytes | 3,256 | 3,256 |

Prompt size reduction: `(15627 - 10898) / 15627 = 30.26%`.

### Selective retry

The oracle intentionally returns an invalid `math` result once. The verifier
rejects it, only `math` is retried, and all 9 final artifacts are verified.
Model calls: 10, which is 1 above the 9-task ideal. The unsuccessful answer is
not stored as a verified dependency.

### Incremental reuse

The cached `math` artifact is invalidated and recomputed. Because the verified
result remains byte-for-byte identical, the other eight verified task records
remain reusable. One call, eight cache hits. If an upstream artifact changes,
its dependent cache fingerprints change and the engine recomputes descendants
as needed (covered by unit tests).

## What is *not* established

- **No private thinking tokens were saved in this demonstration.** The engine
  does not have access to hidden chain-of-thought tokens.
- The reduction of serialized UTF-8 prompt size is not a measurement of actual
  tokenizer token count, model runtime, monetary cost or accuracy.
- Verification in this mock experiment is an exact string comparison; it is
  not proof of general correctness for arbitrary generated code.
- Because the benchmark uses an oracle that ignores its prompt, it cannot
  establish that scoped context preserves real-model quality.
- State SHA-256 checksums are integrity hints; untrusted data can be replaced
  together with its checksum and should not be treated as signed evidence.

## Next rigorous experiment

1. Choose a real model and its **exact tokenizer**. Require provider usage,
   including `reasoning_tokens` when the provider exposes that field.
2. Use an independently verified set of code tasks with deterministic tests.
   Compare full context vs scoped context on identical tasks, paired prompts,
   controlled temperatures, repeated seeds and equal output caps.
3. Report success rate, total input/output/reasoning tokens, model calls,
   compilation/test cost, wall clock time, and API cost. Count retries as cost.
4. Use a held-out dataset and publish failure cases; do not infer accuracy
   from a prompt-length decrease.
5. Train a dedicated model for T0-R if the objective is to reduce its own
   *internal* reasoning length or use latent-state reasoning.
