import argparse, time, numpy as np
try:
    from numba import cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception:
    CUDA_AVAILABLE = False
    cuda = None

def cpu_matmul(A, B):
    return A @ B

if CUDA_AVAILABLE:
    @cuda.jit
    def gpu_matmul_kernel(A, B, C):
        row, col = cuda.grid(2)
        if row < C.shape[0] and col < C.shape[1]:
            tmp = 0.0
            for k in range(A.shape[1]):
                tmp += A[row, k] * B[k, col]
            C[row, col] = tmp

def gpu_matmul(A, B, TPB=16):
    if not CUDA_AVAILABLE:
        raise RuntimeError('CUDA not available')
    C = np.zeros((A.shape[0], B.shape[1]), dtype=np.float32)
    threads = (TPB, TPB)
    blocks = ((C.shape[0]+TPB-1)//TPB, (C.shape[1]+TPB-1)//TPB)
    gpu_matmul_kernel[blocks, threads](A, B, C)
    cuda.synchronize()
    return C

def bench(fn, *args, repeat=3, **kw):
    times = []
    out = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        out = fn(*args, **kw)
        t1 = time.perf_counter()
        times.append(t1 - t0)
    return out, min(times), sum(times)/len(times)

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--n', type=int, default=1024)
    p.add_argument('--repeat', type=int, default=3)
    p.add_argument('--verify', action='store_true', default=True)
    a = p.parse_args()

    N = a.n
    A = np.random.rand(N, N).astype(np.float32)
    B = np.random.rand(N, N).astype(np.float32)

    C_cpu, best_cpu, avg_cpu = bench(cpu_matmul, A, B, repeat=a.repeat)
    print(f'CPU: best={best_cpu*1000:.2f} ms avg={avg_cpu*1000:.2f} ms')

    if CUDA_AVAILABLE:
        C_gpu, best_gpu, avg_gpu = bench(gpu_matmul, A, B, repeat=a.repeat)
        print(f'GPU: best={best_gpu*1000:.2f} ms avg={avg_gpu*1000:.2f} ms')
        if a.verify:
            print('Verify:', np.allclose(C_cpu, C_gpu, atol=1e-2))
    else:
        print('CUDA not available on this system; GPU path skipped.')

if __name__ == '__main__':
    main()
