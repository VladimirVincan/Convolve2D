import cmath
import numpy as np

def omega(p, q):
    return cmath.exp((2.0 * cmath.pi * 1j * q) / p)


def fft_1d(signal):
    n = len(signal)
    if n == 1:
        return signal
    else:
        Feven = fft_1d([signal[i] for i in range(0, n, 2)])
        Fodd = fft_1d([signal[i] for i in range(1, n, 2)])

        combined = [0] * n
        for m in range(n // 2):
            combined[m] = Feven[m] + omega(n, -m) * Fodd[m]
            combined[m + n // 2] = Feven[m] - omega(n, -m) * Fodd[m]

        return combined

def fft_2d(x):
    X_inner = []
    for i in range(x.shape(0)):
        X_inner.append(fft_1d(x[i,:]))


x = np.zeros((5,5))
x[0,0] = 1
print(np.fft.fft2(x))