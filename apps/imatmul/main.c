// Copyright 2020 ETH Zurich and University of Bologna.
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

#include <stdint.h>
#include <string.h>
#include "kernel/imatmul.h"
#include "runtime.h"
#include "util.h"

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

// Define Matrix dimensions:
// C = AB with A=[MxN], B=[NxP], C=[MxP]
extern uint64_t M;
extern uint64_t N;
extern uint64_t P;
extern int64_t a[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int64_t b[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int64_t c[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int64_t c_scalar[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int64_t c_vtmac[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
// Gold results
extern int64_t g[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

// We don't need to redefine MIN as it's already in util.h
// #define MIN(a, b) ((a) < (b) ? (a) : (b))

// Verify the matrix
int verify_matrix(int64_t *result, int64_t *gold, size_t R, size_t C) {
  for (uint64_t i = 0; i < R; ++i) {
    for (uint64_t j = 0; j < C; ++j) {
      uint64_t idx = i * C + j;
      if (result[idx] != gold[idx]) {
        return (i + j) == 0 ? -1 : idx;
      }
    }
  }
  return 0;
}

// Non-vectorized scalar implementation for comparison
void imatmul_scalar(int64_t *c, const int64_t *a, const int64_t *b,
                   const unsigned long int M, const unsigned long int N,
                   const unsigned long int P) {
  // Standard triple-loop matrix multiplication
  for (unsigned long int i = 0; i < M; i++) {
    for (unsigned long int j = 0; j < P; j++) {
      c[i*P + j] = 0;
      for (unsigned long int k = 0; k < N; k++) {
        c[i*P + j] += a[i*N + k] * b[k*P + j];
      }
    }
  }
}

// Custom VTMAC implementation for matrix multiplication
void imatmul_vtmac(int64_t *c, const int64_t *a, const int64_t *b,
                  const unsigned long int M, const unsigned long int N,
                  const unsigned long int P) {
  unsigned long int vl;
  
  // Initialize result matrix to zero
  memset(c, 0, M * P * sizeof(int64_t));
  
  // We work on chunks of the matrix at once
  for (unsigned long int i = 0; i < M; i++) {
    for (unsigned long int j = 0; j < P; j++) {
      // Process the dot product in vector chunks
      for (unsigned long int k = 0; k < N; k += vl) {
        // Set vector length for this chunk
        asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(N - k));
        
        // Load vectors for multiplication
        asm volatile("vle64.v v0, (%0)" ::"r"(&a[i*N + k]));
        asm volatile("vle64.v v4, (%0)" ::"r"(&b[k*P + j]));
        
        // Load current accumulator value
        int64_t current_val = c[i*P + j];
        
        // Set up v8 with current value
        asm volatile("vmv.v.x v8, %0" ::"r"(current_val));
        
        // Execute custom VTMAC instruction
        asm volatile(".insn r 0x57, 0x0, 0x0, v8, v0, v4");
        
        // Extract result to memory and then load it
        volatile int64_t temp_result[4] __attribute__((aligned(32)));
        asm volatile("vse64.v v8, (%0)" ::"r"(temp_result));
        c[i*P + j] = temp_result[0];
      }
    }
  }
}

// Read hardware performance counter for instructions
#ifdef SPIKE
uint64_t read_instret(void) {
  uint64_t instret;
  asm volatile("csrr %0, minstret" : "=r"(instret));
  return instret;
}
#endif

int main() {
  printf("\n");
  printf("=============\n");
  printf("=  IMATMUL  =\n");
  printf("=============\n");
  printf("\n");
  printf("\n");

#ifdef VCD_DUMP
  // Measure only the full-size matmul
  for (uint64_t s = M; s <= M; s *= 2) {
#else
  for (uint64_t s = 4; s <= M; s *= 2) {  // Changed 'int' to 'uint64_t' to match M's type
#endif
    printf("\n");
    printf("------------------------------------------------------------\n");
    printf("Calculating a (%lu x %lu) x (%lu x %lu) matrix multiplication...\n", s,
           s, s, s);
    printf("------------------------------------------------------------\n");
    printf("\n");

    // 1. SCALAR VERSION
    printf("Calculating scalar imatmul...\n");
    
#ifdef SPIKE
    uint64_t scalar_start_inst = read_instret();
#endif

    start_timer();
    imatmul_scalar(c_scalar, a, b, s, s, s);
    stop_timer();
    
#ifdef SPIKE
    uint64_t scalar_end_inst = read_instret();
    uint64_t scalar_inst_count = scalar_end_inst - scalar_start_inst;
#endif

    // Metrics
    int64_t scalar_runtime = get_timer();
    float scalar_performance = 2.0 * s * s * s / scalar_runtime;
    float scalar_utilization = 100 * scalar_performance / (2.0 * NR_LANES);

    printf("The scalar execution took %ld cycles.\n", scalar_runtime);
#ifdef SPIKE
    printf("Scalar instruction count: %lu\n", scalar_inst_count);
#endif
    printf("The scalar performance is %.2f OP/cycle (%.2f%% utilization).\n", 
           scalar_performance, scalar_utilization);
    
    // 2. STANDARD VECTOR VERSION
    printf("Calculating standard vector imatmul...\n");
    
#ifdef SPIKE
    uint64_t vector_start_inst = read_instret();
#endif

    start_timer();
    imatmul(c, a, b, s, s, s);
    stop_timer();
    
#ifdef SPIKE
    uint64_t vector_end_inst = read_instret();
    uint64_t vector_inst_count = vector_end_inst - vector_start_inst;
#endif

    // Metrics
    int64_t vector_runtime = get_timer();
    float vector_performance = 2.0 * s * s * s / vector_runtime;
    float vector_utilization = 100 * vector_performance / (2.0 * NR_LANES);

    printf("The standard vector execution took %ld cycles.\n", vector_runtime);
#ifdef SPIKE
    printf("Standard vector instruction count: %lu\n", vector_inst_count);
#endif
    printf("The standard vector performance is %.2f OP/cycle (%.2f%% utilization).\n", 
           vector_performance, vector_utilization);
           
    // 3. CUSTOM VTMAC VERSION
    printf("Calculating VTMAC imatmul...\n");
    
#ifdef SPIKE
    uint64_t vtmac_start_inst = read_instret();
#endif

    start_timer();
    imatmul_vtmac(c_vtmac, a, b, s, s, s);
    stop_timer();
    
#ifdef SPIKE
    uint64_t vtmac_end_inst = read_instret();
    uint64_t vtmac_inst_count = vtmac_end_inst - vtmac_start_inst;
#endif

    // Metrics
    int64_t vtmac_runtime = get_timer();
    float vtmac_performance = 2.0 * s * s * s / vtmac_runtime;
    float vtmac_utilization = 100 * vtmac_performance / (2.0 * NR_LANES);

    printf("The VTMAC execution took %ld cycles.\n", vtmac_runtime);
#ifdef SPIKE
    printf("VTMAC instruction count: %lu\n", vtmac_inst_count);
#endif
    printf("The VTMAC performance is %.2f OP/cycle (%.2f%% utilization).\n", 
           vtmac_performance, vtmac_utilization);
           
    // COMPARISON METRICS
#ifdef SPIKE
    // Calculate and print the comparison metrics
    printf("\n--- Comparison Metrics ---\n");
    
    // Standard vector vs. scalar
    float vector_scalar_reduction = 100.0 * (scalar_inst_count - vector_inst_count) / scalar_inst_count;
    float vector_scalar_speedup = (float)scalar_runtime / vector_runtime;
    printf("Standard Vector vs Scalar:\n");
    printf("  Instruction reduction: %.2f%% (%lu vs %lu)\n", 
           vector_scalar_reduction, vector_inst_count, scalar_inst_count);
    printf("  Speedup: %.2fx\n", vector_scalar_speedup);
    
    // VTMAC vs. scalar
    float vtmac_scalar_reduction = 100.0 * (scalar_inst_count - vtmac_inst_count) / scalar_inst_count;
    float vtmac_scalar_speedup = (float)scalar_runtime / vtmac_runtime;
    printf("VTMAC vs Scalar:\n");
    printf("  Instruction reduction: %.2f%% (%lu vs %lu)\n", 
           vtmac_scalar_reduction, vtmac_inst_count, scalar_inst_count);
    printf("  Speedup: %.2fx\n", vtmac_scalar_speedup);
    
    // VTMAC vs. standard vector
    float vtmac_vector_reduction = 100.0 * (vector_inst_count - vtmac_inst_count) / vector_inst_count;
    float vtmac_vector_speedup = (float)vector_runtime / vtmac_runtime;
    printf("VTMAC vs Standard Vector:\n");
    printf("  Instruction reduction: %.2f%% (%lu vs %lu)\n", 
           vtmac_vector_reduction, vtmac_inst_count, vector_inst_count);
    printf("  Speedup: %.2fx\n", vtmac_vector_speedup);
#endif

    // Verify the results only for s == M (to keep it simple)
    if (s == M) {
      // Verify the standard vector result
      printf("\nVerifying standard vector result...\n");
      int error = verify_matrix(c, g, s, s);
      if (error != 0) {
        printf("Error code %d\n", error);
        printf("c[%d]=%ld\n", error, c[error]);
        return error;
      } else {
        printf("Standard vector passed.\n");
      }
      
      // Verify the scalar result
      printf("Verifying scalar result...\n");
      error = verify_matrix(c_scalar, g, s, s);
      if (error != 0) {
        printf("Error code %d\n", error);
        printf("c_scalar[%d]=%ld\n", error, c_scalar[error]);
        return error;
      } else {
        printf("Scalar passed.\n");
      }
      
      // Verify the VTMAC result
      printf("Verifying VTMAC result...\n");
      error = verify_matrix(c_vtmac, g, s, s);
      if (error != 0) {
        printf("Error code %d\n", error);
        printf("c_vtmac[%d]=%ld\n", error, c_vtmac[error]);
        return error;
      } else {
        printf("VTMAC passed.\n");
      }
    }
  }

  return 0;
}