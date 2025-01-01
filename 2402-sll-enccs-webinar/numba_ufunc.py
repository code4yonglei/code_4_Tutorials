import math
import numpy as np
import time
import numba


# a simple version without using numba
def f(x, y):
    return math.pow(x, 3.0) + 4*math.sin(y)


# using numba CPU
@numba.vectorize([numba.float64(numba.float64, numba.float64)], target='cpu')
def f_numba_cpu(x, y):
    return math.pow(x,3.0) + 4*math.sin(y)


@numba.vectorize([numba.float64(numba.float64, numba.float64)], target='cuda')
def f_numba_gpu(x,y):
    return math.pow(x,3.0) + 4*math.sin(y)


x = np.random.rand(10000000)
res = np.random.rand(10000000)


## Evaluate performance
start = time.time()
for i in range(10000000):
    res[i]=f(x[i], x[i])
end = time.time()
print(f"\nElapsed time (Without using Numba) = {end - start} s.")

start = time.time()
res = f_numba_cpu(x, x)
end = time.time()
print(f"Elapsed time (Numba for CPU） = {end - start} s.")


# %%timeit -r 1
start = time.time()
res = f_numba_gpu(x, x)
end = time.time()
print(f"Elapsed time (Numba for GPU) = {end - start} s.\n")
