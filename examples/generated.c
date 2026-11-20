/* Generated from T0; no runtime allocations. */
#include <stdint.h>
#include <stddef.h>
#include <math.h>
float t0_run(const float *a0, size_t n) {
  float t0 = 0.0f;
  for (size_t i = 0; i < n; ++i) {
    float x = a0[i];
    if (x > 0.0f) t0 += x * x;
  }
  return t0;
}
