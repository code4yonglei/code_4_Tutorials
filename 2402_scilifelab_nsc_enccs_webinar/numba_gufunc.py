import numpy as np
import numba
import time


def matmul_cpu(A, B, C):
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            tmp=0.0
            for k in range(B.shape[0]):
                tmp += A[i, k] * B[k, j]
            C[i,j] += tmp


#@numba.guvectorize(['(float64[:,:], float64[:,:], float64[:,:])'], '(m,l),(l,n)->(m,n)', target='cpu')
@numba.guvectorize([numba.void(numba.float64[:,:], numba.float64[:,:], numba.float64[:,:])], '(m,l),(l,n)->(m,n)', target='cpu')
def matmul_numba_cpu(A,B,C):
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            tmp=0.0
            for k in range(B.shape[0]):
                tmp += A[i, k] * B[k, j]
            C[i,j] += tmp


#@numba.guvectorize(['(float64[:,:], float64[:,:], float64[:,:])'], '(m,l),(l,n)->(m,n)', target='cuda')
@numba.guvectorize([numba.void(numba.float64[:,:], numba.float64[:,:], numba.float64[:,:])], '(m,l),(l,n)->(m,n)', target='cuda')
def matmul_numba_gpu(A, B, C):
    for i in range(A.shape[0]):
        for j in range(B.shape[1]):
            tmp=0.0
            for k in range(B.shape[0]):
                tmp += A[i, k] * B[k, j]
            C[i,j] += tmp


N = 512
A = np.random.rand(N, N)
B = np.random.rand(N, N)
C = np.random.rand(N, N)


start = time.time()
matmul_cpu(A, B, C)
end = time.time()
print(f"Elapsed time (Without using Numba) = {end - start} s.")


start = time.time()
matmul_numba_cpu(A, B, C)
end = time.time()
print(f"Elapsed time (Numba for CPU) = {end - start} s.")


start = time.time()
matmul_numba_gpu(A, B, C)
end = time.time()
print(f"Elapsed time (Numba for GPU) = {end - start} s.")
