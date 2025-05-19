// vtmac.h
#ifndef VTMAC_H
#define VTMAC_H

#include <stdint.h>

// Initialize test data
void init_vtmac_data(double *a, double *b, unsigned long int n);

// TMAC kernel implementation
void vtmac_kernel(double *dest, double *a, double *b, unsigned long int n);

// Standard matrix multiplication for comparison
void std_matrix_mul(double *dest, double *a, double *b, unsigned long int n);

// Function to verify the results
int vtmac_verify(double *result, double *a, double *b, unsigned long int n);

// Instruction/cycle counting functions
static inline uint64_t read_instret(void) {
  uint64_t instret;
  asm volatile("csrr %0, minstret" : "=r"(instret));
  return instret;
}

static inline uint64_t read_cycles(void) {
  uint64_t cycles;
  asm volatile("csrr %0, mcycle" : "=r"(cycles));
  return cycles;
}

#endif // VTMAC_H