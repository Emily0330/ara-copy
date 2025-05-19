// kernel/ivec8mul.c

// Copyright 2023
//
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "ivec8mul.h"
#include <stddef.h>  // Added for size_t

#define MIN(a, b) ((a) < (b) ? (a) : (b))

void ivec8mul(int8_t *c, const int8_t *a, const int8_t *b, size_t n) {
  // Use the maximum LMUL setting for best throughput
  ivec8mul_m8(c, a, b, n);
}

void ivec8mul_m8(int8_t *c, const int8_t *a, const int8_t *b, size_t n) {
  // Process the vectors using LMUL=8 for maximum throughput
  size_t vl;
  
  // Set the vector configuration - use maximum LMUL (m8) for 8-bit elements
  asm volatile("vsetvli %0, %1, e8, m8, ta, ma" : "=r"(vl) : "r"(n));
  
  // Process the vectors in chunks of vl elements
  for (size_t i = 0; i < n; i += vl) {
    // Adjust vector length for the last iteration if needed
    size_t avl = MIN(n - i, vl);
    asm volatile("vsetvli zero, %0, e8, m8, ta, ma" :: "r"(avl));
    
#ifdef VCD_DUMP
    // Start dumping VCD
    if (i == 256)
      event_trigger = +1;
    // Stop dumping VCD
    if (i == 512)
      event_trigger = -1;
#endif
    
    // Load vectors
    asm volatile("vle8.v v0, (%0)" :: "r"(a + i));
    asm volatile("vle8.v v8, (%0)" :: "r"(b + i));
    
    // Perform element-wise multiplication
    // Note: This will truncate to 8 bits if result overflows
    asm volatile("vmul.vv v16, v0, v8");
    
    // Store the result
    asm volatile("vse8.v v16, (%0)" :: "r"(c + i));
  }
}

int verify_result(int8_t *result, int8_t *gold, size_t n, int8_t threshold) {
  for (size_t i = 0; i < n; i++) {
    int8_t diff = result[i] - gold[i];
    // Take absolute value of diff
    if (diff < 0) diff = -diff;
    
    if (diff > threshold) {
      return i + 1; // Return position of first error (1-indexed)
    }
  }
  return 0; // Success
}
