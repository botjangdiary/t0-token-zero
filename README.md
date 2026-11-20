# T0 v0.2 — AI-native 실행 언어 + 검증형 추론 런타임

**자연어(제한된 규칙) → 타입이 있는 SSA → 복합 의미 연산으로 최적화 → 전용 의미 토큰/압축 바이너리(T0B) → 네이티브 CPU 코드**를 실제로 구현한 작은 연구용 언어입니다.

이 버전에는 **가상의 AI 전용 vocabulary에서 직접 사용할 수 있는 정수형 의미 토큰 스트림**도 구현되어 있습니다. 하지만 **LLM 자체를 학습하지 않았습니다.** 한국어/영어 자연어 변환은 6개 표현 계열만 인식하는 규칙 기반 프런트엔드이며, 범용 자연어 이해를 주장하지 않습니다. `T0B`는 바이너리 코드 압축이지, 현존 LLM의 출력 토큰 수 절감을 자동 보장하지 않습니다. 향후 전용 토크나이저/모델을 결합하는 연구를 위한 하부구조입니다.

## NEW: T0-R v0.2 — 추론 중 *컨텍스트 비용* 최적화

T0 v0.1은 AI가 출력하는 **프로그램 코드**를 압축했습니다. T0-R v0.2는 별도로
**모델에 반복 전달하는 검증된 컨텍스트를 줄이고, 실패한 작업 단계만 재실행**하는 연구용 런타임입니다.
기존 T0B 컴파일러는 그대로 유지됩니다.

```bash
python -m t0 reason-bench
python -m t0 reason-bench --json
python -m pytest -q
```

T0-R는 다음 기능을 제공합니다.

- 의존성 DAG와 위상 정렬: `Step(id, instruction, needs, contract)`.
- 검증된 산출물만 저장: 각 단계는 `validator(...) -> Check`가 통과해야 후속 단계에 전달됩니다.
- 의존성 범위 컨텍스트: 이전 전체 대화 대신 **필요한 선행 단계의 결과만** 프롬프트에 포함합니다.
- 실패 부분 재시도: 검증 실패 시 **해당 단계만 재실행**하고 피드백 길이를 제한합니다.
- 재실행/캐시: 단계와 직접 의존성의 내용 해시가 같으면 검증된 결과를 재사용합니다.
- 예산: 사용자 제공 tokenizer 함수로 입력 상한을 계산하고 출력을 예약합니다. 모델이 제공한 `Usage`를 통해 사용량을 별도로 집계합니다.
- 상태 저장: `State.encode()`/`State.decode()`로 검증된 결과를 JSON에 직렬화합니다 (SHA-256은 오류 감지용이고 서명이 아닙니다).

### T0-R 최소 사용 예제

```python
from t0.reasoning import Graph, Step, Engine, Completion, Check, Usage

graph = Graph([
    Step("spec", "Extract the required behavior"),
    Step("tests", "Write a test case", ("spec",)),
    Step("code", "Implement the requirement", ("spec", "tests")),
])

def model(step, prompt_json, max_output_units):
    # 실제 LLM SDK 호출로 교체할 수 있음. 리턴할 사용량은 PROVIDER 값만 사용.
    return Completion(step.id + " verified artifact")

def validator(step, text, dependencies):
    # 실제로는 pytest, schema validation, 정적 검사 등을 연결.
    return Check(text == step.id + " verified artifact", "artifact invalid")

state, report = Engine(graph, model, validator, goal="Build a utility").run()
print(report.calls, report.reused, report.input_units)
```

**측정 주의:** 기본 토큰 계측기는 `utf8_bytes()`로, **실제 LLM 토큰 수가 아닌 UTF-8 바이트 수**입니다.
진짜 토큰 예산에는 대상 모델의 tokenizer를 `meter=`로 주입하세요.
모델이 `Completion(text, Usage(input_tokens, output_tokens, reasoning_tokens=None))`를 반환하면
런타임은 공급자가 제공한 토큰 사용량을 집계합니다.
`reasoning_tokens`는 공급자가 별도 공개하는 경우에만 측정할 수 있으며 출력 토큰의 **부분집합**입니다.
T0-R는 모델 내부의 비공개 chain-of-thought를 읽거나 압축하지 않습니다.
정확도와 총 비용의 실측 평가는 별도의 실제 모델 테스트가 필요합니다.

### v0.2 재현 가능한 실험 (LLM 호출 없음)

| 항목 | Full-history 기준 | T0-R 의존성 컨텍스트 |
| --- | ---: | ---: |
| 모의 작업 단계 | 9 | 9 |
| 검증된 산출물 수 | 9 | 9 |
| 모델 호출 수 | 9 | 9 |
| 입력 프롬프트 크기 (UTF-8 바이트) | 15,627 | 10,898 |

이 실험에선 프롬프트 **바이트 수**가 30.26% 감소했습니다. 이는 *작업 그래프 + 고정 산출물*을 사용하는 결정적 시뮬레이션으로,
LLM 추론 토큰의 절감률 또는 실제 정확도 개선을 뜻하지 않습니다.
검증 실패 한 번을 고의로 넣으면 **해당 단계만 1회 재시도**하며,
이미 검증한 산출물 하나를 무효화했다가 동일한 값을 다시 검증하면 **1개 단계만 실행하고 8개는 재사용**합니다.
실험 설계 및 상세 한계는 [T0R_RESULTS.md](T0R_RESULTS.md)에 있습니다.

---

## 준비

- Python 3.10 이상 (표준 라이브러리만 사용)
- GCC 또는 Clang (`CC` 환경변수로 선택 가능)
- 테스트 실행용 `pytest` (선택)
- Linux에서 컴파일된 공유 라이브러리 `.so`를 `ctypes`로 호출합니다. macOS에서는 `.dylib` 등 추가 조정이 필요합니다.

프로젝트 디렉터리에서 `python -m t0 ...` 형식으로 실행합니다. 설치하고 싶다면 `pip install -e .` 후 `t0 ...`로 실행할 수 있습니다.

## 바로 실행

```bash
# 가장 간단한 자연어 → 실행 통합 명령
python -m t0 ask '양수만 제곱해서 합산해줘' --data '1,-2,3' --show-ir --show-tokens

# 한국어 → T0 텍스트 + 최적화된 바이너리
python -m t0 nl '실수 배열에서 양수만 제곱한 후 전부 더해줘' \
  --output /tmp/example.t0 --pack /tmp/example.t0b

# 사람이 볼 수 있게 디스어셈블 (최적화된 rp2 opcode가 표시됨)
python -m t0 dump /tmp/example.t0b

# 네이티브 코드 실행: 1² + 3² = 10
python -m t0 run /tmp/example.t0b --data '1,-2,3' --backend native

# 비교 검증
python -m t0 run /tmp/example.t0b --data '1,-2,3' --backend reference

# 의미 토큰 ID 확인 (전용 vocabulary에선 5토큰)
python -m t0 tokens /tmp/example.t0b
# [16, 69, 128, 3, 1]

# 토큰을 다시 타입 검증된 T0B로 변환
python -m t0 detokenize '[16,69,128,3,1]' -o /tmp/roundtrip.t0b

# C11 소스 및 네이티브 .so 생성
python -m t0 compile /tmp/example.t0b -o /tmp/t0.so --emit-c /tmp/t0.c

# 마이크로벤치마크
python bench.py --size 200000 --trials 9

# 테스트
python -m pytest -q
```

## 지원 자연어 예제

| 자연어 | 최적화된 T0 | 반환형 |
| --- | --- | --- |
| 양수만 제곱해서 합해줘 | `rp2(@0)` | f32 |
| 두 배열의 내적 | `dot(@0,@1)` | f32 |
| 양수 개수 | `countp(@0)` | u64 |
| 배열 합계 | `sum(@0)` | f32 |
| 배열 최댓값 | `max(@0)` | f32 |
| 배열 평균 | `mean(@0)` | f32 |

미지원 문장, 대표적인 부정·제외 표현 등은 **추측하지 않고 오류를 반환**하도록 했습니다. 다만 이 작은 패턴 매칭기는 모든 종류의 모호성을 탐지하지는 못하므로 실제 배포에는 별도의 의미 검증이 필수입니다.

## 최소 언어 예제

입력 소스 `examples/positive_square_sum.t0`:

```text
F0(f32[*])->f32!p~strict{
  %0=gt(@0,0.0f)
  %1=mul(@0,@0)
  %2=sel(%0,%1,0.0f)
  %3=sum(%2)
  R=%3
}
```

타입 검사 및 합성 패턴 매칭 이후:

```text
F0(f32[*])->f32!p~strict{%0=rp2(@0);R=%0}
```

`!p`는 순수 계산을 의미합니다. `~strict`는 binary32 누적 순서를 보존하고 fast-math를 금지하는 모드입니다. `~fast`는 `-ffast-math`를 허용하므로 수치 결과가 바뀔 수 있습니다. `@0`은 입력 배열이고 `%0`은 SSA 결과 참조입니다.

## 설계 원칙

- **검증**: IR의 순서, 타입, 매개변수 개수, 반환 타입, 연산 개수 제한을 확인합니다. 바이너리 로드 시 CRC32도 검사합니다. CRC32는 악의적 변조를 막는 암호학적 서명이 아닙니다.
- **제로 중간 배열**: `gt`, `mul`, `sel`로 이루어진 배열 계산은 reduce 루프 안에 인라인되어 중간 배열을 할당하지 않습니다.
- **복합 연산**: `sum(sel(gt(x,0),mul(x,x),0))` → `rp2(x)`, `sum(mul(x,y))` → `dot(x,y)` 를 의미 보존 조건에 맞게 변환합니다.
- **타입**: 외부 인자는 `f32[*]`, 반환은 `f32` 또는 `u64`, 내부 값은 스칼라/벡터 `f32` 및 `bool`입니다.
- **실행**: 범용 CPU에 대해 C11을 생성한 뒤 `-O3`로 GCC/Clang을 사용합니다. 함수의 C ABI는 `float t0_run(const float *a0, size_t n)` 또는 다중 입력/`uint64_t` 변형입니다. 모든 벡터 입력은 동일 길이여야 합니다.
- **숫자 계약**: 값과 연산은 IEEE 754 binary32, 합산은 순차 순서. `mean`/`max`에 빈 배열은 허용하지 않습니다. 일반적인 Python `float`는 binary64이므로 출력의 마지막 자릿수가 다를 수 있습니다.

## AI 전용 의미 토큰 (실제로 인코딩/디코딩 가능)

T0에서 `rp2(@0)` 하나를 나타내는 의미 토큰 스트림은 다음과 같습니다.

```text
[16, 69, 128, 3, 1]
  │   │    │  │  └─ EOS
  │   │    │  └──── RET_LAST
  │   │    └─────── ARG0
  │   └──────────── OP_RP2
  └──────────────── SIG_1_F32_STRICT
```

비압축 SSA에서는 동일 계산이 **15개의 의미 토큰**, 복합 opcode 최적화 이후에는 **5개의 의미 토큰**입니다. opcode 하나당 하나의 토큰을 쓰는 **T0 전용 모델의 어휘**를 가정한 측정이며, GPT 또는 다른 상용 모델에 그대로 출력하면 5토큰이라는 뜻이 아닙니다. `t0/semantic_tokens.py`의 디코더는 토큰 스트림을 실제 검증 가능한 프로그램으로 복원합니다.

## T0B 바이너리 형식 (v1)

```text
Magic("T0B\\0") | Version(u8) | FPMode(u8)
| FnID(ULEB128) | Arity(ULEB128) | ArgType[u8] * N
| ReturnType(u8) | NodeCount(ULEB128)
| Nodes[ Opcode(u8) + Operands(Ref) ]
| ReturnRef | CRC32(u32 LE)

Ref = ULEB128((index << 2) | tag)
tag 0: @input_index    tag 1: %temporary_index
tag 2: inline float32 LE (tag value is precisely 2)
```

현재 T0B의 `rp2`는 **1바이트 opcode**로 저장됩니다. T0B에는 매직, 타입, 함수 정의 및 체크섬 등의 추가 데이터가 있으며, 따라서 1바이트짜리 완전한 프로그램이라는 뜻은 아닙니다.

## 속도 해석 시 주의

`bench.py`는 준비한 배열을 네이티브 함수에 전달한 뒤 루프 실행 시간만 측정합니다. 자연어 변환·프로그램 검증·C 코드 생성·네이티브 컴파일·Python→C 입력 복사 비용은 **포함하지 않습니다**. Python 직접 반복문과 비교한 수치는 실행 전략의 차이를 보여주는 실험이지, 다른 최적화 컴파일러나 NumPy, CUDA에 대한 우월성을 입증하지는 않습니다. 또한 이 버전은 GPU 백엔드와 범용 제어 흐름이 아직 없습니다.

## 다음 연구 단계

1. 실제 자연어 → T0 데이터셋과 DSL constrained decoder
2. 문법 토큰 전용 어휘 훈련 및 실제 LLM 출력 토큰 수 측정
3. 빈번한 SSA 부분 그래프를 자동 추출하는 복합 연산 사전
4. MLIR/GPU 백엔드 및 하드웨어별 자동 튜닝
5. 정형 의미 동치 검증과 메모리 안전성/권한 모델

## 프로젝트 구조

```text
t0/ir.py          typed SSA, verifier
t0/frontend.py    text assembler, limited Korean/English mapping
t0/optimizer.py   composite opcode rewrite
t0/binary.py      T0B encoder/decoder
t0/semantic_tokens.py  dedicated semantic vocabulary encoder/decoder
t0/backend.py     loop fusion, C11 emitter, native compiler
t0/reference.py   Python reference interpreter
t0/__main__.py    CLI
examples/         sample .t0 programs
tests/            unit and native integration tests
bench.py          repeatable local microbenchmark
```

MIT 라이선스로 자유롭게 수정·실험할 수 있습니다.
