#!/usr/bin/env python3
# script/gen_data.py

# Copyright 2023
#
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Generate input data for 8-bit vector multiplication
# arg: vector length

import numpy as np
import sys

def emit(name, array, alignment='8'):
  print(".global %s" % name)
  print(".balign " + alignment)
  print("%s:" % name)
  bs = array.tobytes()
  for i in range(0, len(bs), 4):
    s = ""
    for n in range(4):
      if i+3-n < len(bs):
        s += "%02x" % bs[i+3-n]
      else:
        s += "00"
    print("    .word 0x%s" % s)

# Vector length
if len(sys.argv) > 1:
  N = int(sys.argv[1])
else:
  # Default vector length
  N = 1024

# Generate random 8-bit integers from -128 to 127
np.random.seed(42)  # For reproducibility
a = np.random.randint(-128, 128, N, dtype=np.int8)
b = np.random.randint(-128, 128, N, dtype=np.int8)

# Calculate the golden vector - note that 8-bit multiplication wraps on overflow
g = np.multiply(a, b).astype(np.int8)

# Empty result vector
c = np.zeros(N, dtype=np.int8)

# Print information to file
print(".section .data,\"aw\",@progbits")
emit("N", np.array(N, dtype=np.uint64))
emit("a", a, 'NR_LANES*4')
emit("b", b, 'NR_LANES*4')
emit("c", c, 'NR_LANES*4')
emit("g", g, 'NR_LANES*4')