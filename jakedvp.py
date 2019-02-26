import numpy as np
import timeit

# The module function timeit.timeit(stmt, setup, timer, number) accepts four arguments:
# 1) stmt which is the statement you want to measure; it defaults to ‘pass’.
# 2) setup which is the code that you run before running the stmt; it defaults to ‘pass’.
# We generally use this to import the required modules for our code.
# 3) timer which is a timeit.Timer object; it usually has a sensible default value so you don’t have to worry about it.
# 4) number which is the number of executions you’d like to run the stmt.

# The module function timeit.repeat repeats the execution of timeit.timeit a defined number of times.

def DFT_slow(x):
    """Compute the discrete Fourier Transform of the 1D array x"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]
    n = np.arange(N)
    k = n.reshape((N, 1))
    M = np.exp(-2j * np.pi * k * n / N)
    return np.dot(M, x)


def FFT(x):
    """A recursive implementation of the 1D Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if N % 2 > 0:
        raise ValueError("size of x must be a power of 2")
    elif N <= 32:  # this cutoff should be optimized
        return DFT_slow(x)
    else:
        X_even = FFT(x[::2])
        X_odd = FFT(x[1::2])
        factor = np.exp(-2j * np.pi * np.arange(N) / N)
        return np.concatenate([X_even + factor[:N // 2] * X_odd, X_even + factor[N // 2:] * X_odd])


def FFT_vectorized(x):
    """A vectorized, non-recursive version of the Cooley-Tukey FFT"""
    x = np.asarray(x, dtype=float)
    N = x.shape[0]

    if np.log2(N) % 1 > 0:
        raise ValueError("size of x must be a power of 2")

    # N_min here is equivalent to the stopping condition above,
    # and should be a power of 2
    N_min = min(N, 32)

    # Perform an O[N^2] DFT on all length-N_min sub-problems at once
    n = np.arange(N_min)
    k = n[:, None]
    M = np.exp(-2j * np.pi * n * k / N_min)
    X = np.dot(M, x.reshape((N_min, -1)))

    # build-up each level of the recursive calculation all at once
    while X.shape[0] < N:
        X_even = X[:, :X.shape[1] // 2]
        X_odd = X[:, X.shape[1] // 2:]
        factor = np.exp(-1j * np.pi * np.arange(X.shape[0])
                        / X.shape[0])[:, None]
        X = np.vstack([X_even + factor * X_odd,
                       X_even - factor * X_odd])

    return X.ravel()


def DFT_slow_time():
    SETUP_CODE = ''' 
from __main__ import DFT_slow
import numpy as np'''
    TEST_CODE = ''' 
x = np.random.random(1024)
DFT_slow(x)'''
    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=3,
                          number=10)
    print('DFT_slow(x) time: {}'.format(min(times)))


def FFT_time():
    SETUP_CODE = ''' 
from __main__ import FFT
import numpy as np'''
    TEST_CODE = ''' 
x = np.random.random(1024)
FFT(x)'''
    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=3,
                          number=10)
    print('FFT(x) time: {}'.format(min(times)))


def FFT_vectorized_time():
    SETUP_CODE = ''' 
from __main__ import FFT_vectorized
import numpy as np'''
    TEST_CODE = ''' 
x = np.random.random(1024)
FFT_vectorized(x)'''
    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=3,
                          number=10)
    print('FFT_vectorized(x) time: {}'.format(min(times)))


def np_fft_time():
    SETUP_CODE = ''' 
import numpy as np'''
    TEST_CODE = ''' 
x = np.random.random(1024)
np.fft.fft(x)'''
    times = timeit.repeat(setup=SETUP_CODE,
                          stmt=TEST_CODE,
                          repeat=3,
                          number=10)
    print('np.fft.fft(x) time: {}'.format(min(times)))


x = np.random.random(1024)
print(np.allclose(DFT_slow(x), np.fft.fft(x)))
print(np.allclose(FFT(x), np.fft.fft(x)))
print(np.allclose(FFT_vectorized(x), np.fft.fft(x)))

DFT_slow_time()
FFT_time()
FFT_vectorized_time()
np_fft_time()
