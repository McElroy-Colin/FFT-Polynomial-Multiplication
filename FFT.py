# FFT and inverse FFT implementation functions.

import cmath

def _root_of_unity(n: int, k: int) -> complex:
    """
    Returns omega_n^k, the k-th of the n-th roots of unity.
    n and k should be integers, with n > 0.
    """

    return cmath.exp(-2j * cmath.pi * k / n)


def fft(coeffs: list[int | float | complex]) -> list[complex]:
    """
    Computes the FFT of the coefficient list `coeffs`.
    len(coeffs) must be a power of 2.
    Returns a list of the complex FFT coefficients at the n-th roots of unity.
        i.e. result[k] = A(omega_n^k).
    """

    n = len(coeffs)

    # Base case: FFT of a single coefficient is itself
    if n == 1:
        return coeffs
    # Ensure that the length of coeffs is a power of 2.
    elif n & (n - 1) != 0:
        raise ValueError("fft(), length of coeffs must be a power of 2")

    # Split into even-indexed and odd-indexed coefficients
    even = fft(coeffs[0::2])
    odd  = fft(coeffs[1::2])

    result = [0] * n
    # Standard FFT formula, compute two coefficients per iteration.
    for k in range(n // 2):
        w = _root_of_unity(n, k)
        result[k] = even[k] + w * odd[k]
        result[k + n // 2] = even[k] - w * odd[k]

    return result


def _ifft_recursive(values: list[int | float | complex]) -> list[complex]:
    """
    Recursive helper for the inverse FFT. Uses the conjugate root of unity instead.
    Does NOT divide by n as this is a helper, normalization occurs after recursion.
    """

    # Good on recursion, Python len() is O(1).
    n = len(values)

    # Base case.
    if n == 1:
        return values

    # Recursively pass even and odd indexed values to the inverse FFT.
    even = _ifft_recursive(values[0::2])
    odd  = _ifft_recursive(values[1::2])

    result = [0] * n
    # Standard IFFT formula.
    for k in range(n // 2):
        w = _root_of_unity(n, -k)   # -k since inverse uses the conjugate root of unity.
        result[k] = even[k] + w * odd[k]
        result[k + n // 2] = even[k] - w * odd[k]

    return result


def ifft(values: list[int | float | complex]) -> list[complex]:
    """
    Computes the inverse FFT of `values`.
    len(values) must be a power of 2.
    Returns a list of complex numbers: the coefficients of the polynomial
    whose evaluations at the n-th roots of unity are `values`.
    """

    n = len(values)
    # Ensure that the length of values is a power of 2.
    if n & (n - 1) != 0:
        raise ValueError("ifft(), length of values must be a power of 2")
    
    raw = _ifft_recursive(values)
    # Normalize each coefficient once after recursion.
    return [x / n for x in raw]