// main.c
#include <stdint.h>
#include <string.h>
#include "kernel/vtmac.h"
#include "runtime.h"
#include "util.h"
#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

#define TEST_SIZE 1024

// Reference to data arrays initialized in data.S
extern uint64_t N;
extern double a[] __attribute__((aligned(4 * NR_LANES), section(".l2")));
extern double b[] __attribute__((aligned(4 * NR_LANES), section(".l2")));
extern double result[] __attribute__((aligned(4 * NR_LANES), section(".l2")));
extern double std_result[] __attribute__((aligned(4 * NR_LANES), section(".l2")));

#define VERIFY 1

int main() {
  printf("\n");
  printf("=========================================\n");
  printf("= VTMAC vs Standard Matrix Multiplication =\n");
  printf("=========================================\n");
  printf("\n");
  
  for (int s = 4; s <= TEST_SIZE; s *= 2) {
    printf("\n");
    printf("-------------------------------------------------------\n");
    printf("Testing with vector size %d...\n", s);
    printf("-------------------------------------------------------\n");
    printf("\n");
    
    // Initialize data
    printf("Initializing test data...\n");
    init_vtmac_data(a, b, s);
    
    // Run standard matrix multiplication
    printf("Running standard matrix multiplication...\n");
    uint64_t std_instr_start = read_instret();
    uint64_t std_cycle_start = read_cycles();
    
    std_matrix_mul(std_result, a, b, s);
    
    uint64_t std_cycle_end = read_cycles();
    uint64_t std_instr_end = read_instret();
    
    uint64_t std_instr_count = std_instr_end - std_instr_start;
    uint64_t std_cycle_count = std_cycle_end - std_cycle_start;
    
    // Run TMAC kernel
    printf("Running TMAC kernel...\n");
    uint64_t tmac_instr_start = read_instret();
    uint64_t tmac_cycle_start = read_cycles();
    
    vtmac_kernel(result, a, b, s);
    
    uint64_t tmac_cycle_end = read_cycles();
    uint64_t tmac_instr_end = read_instret();
    
    uint64_t tmac_instr_count = tmac_instr_end - tmac_instr_start;
    uint64_t tmac_cycle_count = tmac_cycle_end - tmac_cycle_start;
    
    // Print performance metrics
    printf("\nPerformance Comparison for size %d:\n", s);
    printf("Standard Matrix Multiplication:\n");
    printf("  Instructions: %llu\n", (unsigned long long)std_instr_count);
    printf("  Cycles: %llu\n", (unsigned long long)std_cycle_count);
    printf("  Cycles per element: %.2f\n", (double)std_cycle_count / s);
    
    printf("TMAC Matrix Multiplication:\n");
    printf("  Instructions: %llu\n", (unsigned long long)tmac_instr_count);
    printf("  Cycles: %llu\n", (unsigned long long)tmac_cycle_count);
    printf("  Cycles per element: %.2f\n", (double)tmac_cycle_count / s);
    
    printf("Improvement Ratio:\n");
    printf("  Instruction count: %.2fx\n", (double)std_instr_count / tmac_instr_count);
    printf("  Cycle count: %.2fx\n", (double)std_cycle_count / tmac_cycle_count);
    
    // Verify the result
    // 在 main.c 中
    // 驗證結果
    if (VERIFY) {
      printf("Verifying results...\n");
      if (vtmac_verify(result, a, b, s)) {
        printf("Verification failed!\n");
        return 1;
      } else {
        printf("Verification passed.\n");
      }
    }
  }
  
  printf("Done!\n");
  return 0;
}