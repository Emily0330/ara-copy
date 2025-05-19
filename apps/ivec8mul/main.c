// main.c
#include <stdint.h>
#include <stddef.h>
#include <string.h>

#include "kernel/ivec8mul.h"
#include "runtime.h"
#include "util.h"

#ifdef SPIKE
#include <stdio.h>
#elif defined ARA_LINUX
#include <stdio.h>
#else
#include "printf.h"
#endif

// Define vector size
extern uint64_t N;

// Input vectors
extern int8_t a[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t b[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
extern int8_t c[] __attribute__((aligned(32 * NR_LANES), section(".l2")));
// Gold results
extern int8_t g[] __attribute__((aligned(32 * NR_LANES), section(".l2")));

#define THRESHOLD 0  // exact match for integer multiplication

// 使用RISC-V CSR直接訪問性能計數器
static inline uint64_t read_cycles() {
    uint64_t cycles;
    asm volatile("csrr %0, cycle" : "=r"(cycles));
    return cycles;
}

static inline uint64_t read_instret() {
    uint64_t instret;
    asm volatile("csrr %0, instret" : "=r"(instret));
    return instret;
}

// 標量版本的實現
void ivec8mul_scalar(int8_t *c, const int8_t *a, const int8_t *b, size_t n) {
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i] * b[i];
    }
}

// 性能測量結構
typedef struct {
    uint64_t cycles;
    uint64_t instructions;
} perf_count_t;

// 測量函數執行的性能
void measure_performance(void (*func)(int8_t*, const int8_t*, const int8_t*, size_t), 
                         const char* func_name, perf_count_t* counts) {
    uint64_t start_cycles, end_cycles;
    uint64_t start_instret, end_instret;
    
    // 記錄開始時的計數器值
    start_cycles = read_cycles();
    start_instret = read_instret();
    
    // 執行目標函數
    (*func)(c, a, b, N);
    
    // 記錄結束時的計數器值
    end_cycles = read_cycles();
    end_instret = read_instret();
    
    // 計算差值
    counts->cycles = end_cycles - start_cycles;
    counts->instructions = end_instret - start_instret;
    
    // 輸出結果
    printf("\n=== %s Performance ===\n", func_name);
    printf("Cycles:        %lu\n", counts->cycles);
    printf("Instructions:  %lu\n", counts->instructions);
    printf("IPC:           %.2f\n", (double)counts->instructions / counts->cycles);
}

int main() {
    perf_count_t vector_counts, scalar_counts;
    
    printf("\n");
    printf("=============\n");
    printf("=  IVEC8MUL =\n");
    printf("=============\n");
    printf("\n");
    
    printf("Vector Length: %ld\n", N);
    printf("Calculating 8-bit vector multiplication...\n");
    
    // 測量向量版本
    memset(c, 0, N * sizeof(int8_t));
    measure_performance(ivec8mul, "Vector Implementation", &vector_counts);
    
    // 驗證向量結果
    int vector_error = verify_result(c, g, N, THRESHOLD);
    if (vector_error != 0) {
        printf("Vector Implementation: Error at position %d: c[%d]=%d, g[%d]=%d\n", 
               vector_error-1, vector_error-1, c[vector_error-1], 
               vector_error-1, g[vector_error-1]);
    } else {
        printf("Vector Implementation: Passed. All results match the expected values.\n");
    }
    
    // 測量標量版本
    memset(c, 0, N * sizeof(int8_t));
    measure_performance(ivec8mul_scalar, "Scalar Implementation", &scalar_counts);
    
    // 驗證標量結果
    int scalar_error = verify_result(c, g, N, THRESHOLD);
    if (scalar_error != 0) {
        printf("Scalar Implementation: Error at position %d: c[%d]=%d, g[%d]=%d\n", 
               scalar_error-1, scalar_error-1, c[scalar_error-1], 
               scalar_error-1, g[scalar_error-1]);
    } else {
        printf("Scalar Implementation: Passed. All results match the expected values.\n");
    }
    
    // 比較性能
    printf("\n=== Performance Comparison ===\n");
    printf("Instruction Reduction: %.2fx\n", 
           (double)scalar_counts.instructions / vector_counts.instructions);
    printf("Speedup (Cycles): %.2fx\n", 
           (double)scalar_counts.cycles / vector_counts.cycles);
    
    return 0;
}