# Function to test our FFT and IFFT implementations against numpy's implementation.

from FFT import fft, ifft
from utils import random_polynomial
import numpy as np


def _compare_with_numpy(coeffs, tol=1e-9):
    """
    Runs our fft and numpy's fft on the same coeffs, checks they match.
    """
    ours = np.array(fft(coeffs))
    theirs = np.fft.fft(coeffs)
    return np.allclose(ours, theirs, atol=tol)

def _compare_ifft_with_numpy(values, tol=1e-9):
    """
    Runs our ifft and numpy's ifft on the same values, checks they match.
    """
    ours = np.array(ifft(values))
    theirs = np.fft.ifft(values)

    return np.allclose(ours, theirs, atol=tol)


def run_tests(num_tests=100, max_degree=15, verbose=False):
    passed = 0
    for i in range(num_tests):
        coeffs = random_polynomial(max_degree=max_degree)
        fft_ok = _compare_with_numpy(coeffs)
        ifft_ok = _compare_ifft_with_numpy(fft(coeffs))

        if verbose or not (fft_ok and ifft_ok):
            print(f"Test {i + 1}: fft coeffs={coeffs}  ->  {'PASS' if (fft_ok and ifft_ok) else 'FAIL'}")

        if fft_ok:
            passed += 1
        else:
            print(f"  ours:   {fft(coeffs)}")
            print(f"  numpy:  {np.fft.fft(coeffs)}")

        if not ifft_ok:
            print(f"  ours:   {ifft(coeffs)}")
            print(f"  numpy:  {np.fft.ifft(coeffs)}")

    print(f"\n{passed}/{num_tests} tests passed")