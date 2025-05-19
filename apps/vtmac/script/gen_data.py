#!/usr/bin/env python3
# Copyright 2025 [Your Organization]
#
# SPDX-License-Identifier: Apache-2.0
import random
import numpy as np
import sys

def emit(name, array, alignment='8'):
    print(".global %s" % name)
    print(".balign " + alignment)
    print("%s:" % name)
    bs = array.tobytes()
    for i in range(0, len(bs), 4):
        s = ""
        for n in range(min(4, len(bs) - i)):
            s += "%02x" % bs[i+min(3, len(bs)-i-1)-n]
        print(" .word 0x%s" % s)

############
## SCRIPT ##
############
# Default value if no argument is provided
n = 1024
if len(sys.argv) == 2:
    n = int(sys.argv[1])

# Create test arrays
a = np.random.rand(n).astype(np.float64)
b = np.random.rand(n).astype(np.float64)
result = np.zeros(n).astype(np.float64)  # For TMAC result
std_result = np.zeros(n).astype(np.float64)  # For standard result

# Create the file
print(".section .data,\"aw\",@progbits")
print(".global N")
print(".section .sdata,\"aw\",@progbits")
print("N: .dword %d" % n)

print(".section .l2,\"aw\",@progbits")
emit("a", a, 'NR_LANES*4')
emit("b", b, 'NR_LANES*4')
emit("result", result, 'NR_LANES*4')
emit("std_result", std_result, 'NR_LANES*4')