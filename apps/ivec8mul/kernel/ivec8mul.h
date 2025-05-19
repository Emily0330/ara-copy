// kernel/ivec8mul.h

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

#ifndef IVEC8MUL_H
#define IVEC8MUL_H

#include <stdint.h>
#include <stddef.h>  // Added for size_t

// Function to perform element-wise vector multiplication of 8-bit integers
void ivec8mul(int8_t *c, const int8_t *a, const int8_t *b, size_t n);

// Implementation using maximum LMUL setting for best performance
void ivec8mul_m8(int8_t *c, const int8_t *a, const int8_t *b, size_t n);

// Function to verify the results
int verify_result(int8_t *result, int8_t *gold, size_t n, int8_t threshold);

void ivec8mul_scalar(int8_t *c, const int8_t *a, const int8_t *b, size_t n);

// For VCD dumping
extern int64_t event_trigger;

#endif