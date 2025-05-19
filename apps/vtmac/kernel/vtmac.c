// vtmac.c
#include "vtmac.h"
#include "runtime.h"
#include "util.h"
#include <math.h>
#include <stdint.h>
#include <string.h>
#ifndef SPIKE
#include "printf.h"
#else
#include <stdio.h>
#endif

// Initialize test data
void init_vtmac_data(double *a, double *b, unsigned long int n) {
  for (unsigned long int i = 0; i < n; i++) {
    // Initialize with some pattern
    a[i] = (double)i * 0.5;
    b[i] = (double)i * 0.25;
  }
}

// TMAC kernel using custom instruction
void vtmac_kernel(double *dest, double *a, double *b, unsigned long int n) {
  unsigned long int vl;
  asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(n));
  
  for (unsigned long int i = 0; i < n; i += vl) {
    // Set vector length for this iteration
    asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(n - i));
    
    // Load vectors
    asm volatile("vle64.v v0, (%0)" ::"r"(a + i));
    asm volatile("vle64.v v4, (%0)" ::"r"(b + i));
    
    // Custom TMAC instruction for matrix multiplication
    asm volatile(".insn r 0x57, 0x0, 0x39, v8, v0, v4");
    
    // Store results
    asm volatile("vse64.v v8, (%0)" ::"r"(dest + i));
  }
}

// Standard matrix multiplication using regular RISC-V vector instructions
void std_matrix_mul(double *dest, double *a, double *b, unsigned long int n) {
  unsigned long int vl;
  asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(n));
  
  for (unsigned long int i = 0; i < n; i += vl) {
    // Set vector length for this iteration
    asm volatile("vsetvli %0, %1, e64, m4, ta, ma" : "=r"(vl) : "r"(n - i));
    
    // Load vectors
    asm volatile("vle64.v v0, (%0)" ::"r"(a + i));
    asm volatile("vle64.v v4, (%0)" ::"r"(b + i));
    
    // Standard vector multiply
    asm volatile("vfmul.vv v8, v0, v4");
    
    // Store results
    asm volatile("vse64.v v8, (%0)" ::"r"(dest + i));
  }
}

// 修正後的函數簽名，與 vtmac.h 中的聲明一致
int vtmac_verify(double *result, double *a, double *b, unsigned long int n) {
  int errors = 0;
  
  for (unsigned long int i = 0; i < n; i++) {
    // 標準矩陣乘法計算預期結果
    double std_result = a[i] * b[i];
    
    // 與預期結果比較
    if (fabs(result[i] - std_result) > 1e-2) {
      printf("Error at index %ld: got %f, expected %f\n",
             i, result[i], std_result);
      errors++;
      if (errors > 10) {
        printf("Too many errors, stopping verification.\n");
        return 1;
      }
    }
  }
  
  return (errors > 0) ? 1 : 0;
}