# T0 v0.1 — 실행 검증 기록

검증 환경: Linux x86_64, Python 3.13.5, GCC 14.2.0 (사용 가능 GCC, Clang). 이 결과는 해당 실험 환경에서만 확인한 수치입니다.

## E2E 데모

```bash
python -m t0 ask '양수만 제곱해서 합산해줘' --data '1,-2,3' --show-ir --show-tokens
```

출력:

```text
F0(f32[*])->f32!p~strict{
  %0=rp2(@0)
  R=%0
}
Semantic tokens: [16, 69, 128, 3, 1]
Result: 10.0
```

## 토큰 및 압축

| 표현 | 최적화 전 | 최적화 후 |
|---|---:|---:|
| T0 텍스트(UTF-8) | 89바이트 | 41바이트 |
| T0B 바이너리 | 36바이트 | 18바이트 |
| 전용 semantic vocabulary 토큰 ID | 15개 | 5개 |

**중요:** 토큰 수는 이 프로젝트가 제안한 전용 vocabulary에 대한 수치입니다. GPT-6 등 기존 LLM의 토크나이저에서 같은 수치를 보장하지 않으며, 실제 AI 토큰·비용 개선은 전용 모델 연결 이후 측정해야 합니다.

## 실행 성능 (한 차례 실험의 관측치)

```text
Input: 200,000 f32 elements; 9 trials (median)
Python loop: 7.696 ms
T0 native: 1.034 ms
Speed ratio: 7.4x
```

- 네이티브 실행 준비가 끝난 함수에 대한 호출 시간입니다. 파싱, 컴파일, ctypes 배열 준비 및 데이터 복사 비용을 측정에서 제외합니다.
- Python 비교는 binary64 산술, T0는 binary32 산술입니다. 실행 결과의 미세한 차이는 의도된 정밀도 차이입니다.
- NumPy / Numba / Rust / C++ / 다른 최적화 컴파일러에 비해 빠르다는 의미는 **아닙니다**.
- 벤치마크 변동성 때문에 성능 수치를 일반적인 보장으로 해석하면 안 됩니다. `python bench.py --size 200000 --trials 9`로 재측정할 수 있습니다.

## 자동 테스트

```text
28 passed in 1.13s
```

T0 IR 검사, 텍스트/바이너리/semantic token 왕복, 무효 참조 및 바이트 손상 감지, CPU 네이티브 실행 및 참조 실행의 일치성, 샘플 자연어 변환 등을 검사했습니다.
