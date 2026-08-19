# FFT and inverse FFT implementation functions.

import cmath

from src.utils import pad_match_lsts, clean_coefficients


def _root_of_unity(n: int, k: int) -> complex:
    """
    Returns omega_n^k, the k-th of the n-th roots of unity.
    n and k should be integers, with n > 0.
    """

    return cmath.exp(-2j * cmath.pi * k / n)


def fft(coeffs: list[int | float | complex]) -> tuple[list[complex], int]:
    """
    Computes the FFT of the coefficient list `coeffs`.
    len(coeffs) must be a power of 2.

    Returns a tuple:
        - result: list of complex FFT coefficients at the n-th roots of unity,
          i.e. result[k] = A(omega_n^k).
        - mults: number of complex multiplications performed (twiddle factor
          multiplications only; root-of-unity computation itself doesn't count,
          since those are treated as precomputed constants).
    """

    n = len(coeffs)

    # Base case: FFT of a single coefficient is itself, no multiplications needed
    if n == 1:
        return coeffs, 0
    # Ensure that the length of coeffs is a power of 2.
    elif n == 0 or (n & (n - 1) != 0):
        raise ValueError("fft(), length of coeffs must be a power of 2")

    # Split into even-indexed and odd-indexed coefficients
    even, even_mults = fft(coeffs[0::2])
    odd, odd_mults = fft(coeffs[1::2])

    result = [0] * n
    mults = even_mults + odd_mults

    # Standard FFT formula, compute two coefficients per iteration.
    for k in range(n // 2):
        w = _root_of_unity(n, k)
        product = w * odd[k]
        mults += 1

        result[k] = even[k] + product
        result[k + n // 2] = even[k] - product

    return result, mults


def _ifft_recursive(values: list[int | float | complex]) -> tuple[list[complex], int]:
    """
    Recursive helper for the inverse FFT. Uses the conjugate root of unity instead.
    Does NOT divide by n as this is a helper, normalization occurs after recursion.

    Returns a tuple:
        - result: the un-normalized recursive IFFT values.
        - mults: number of complex multiplications performed (twiddle factor
          multiplications only, same convention as fft()).
    """

    # Good on recursion, Python len() is O(1).
    n = len(values)

    # Base case.
    if n == 1:
        return values, 0

    # Recursively pass even and odd indexed values to the inverse FFT.
    even, even_mults = _ifft_recursive(values[0::2])
    odd, odd_mults = _ifft_recursive(values[1::2])

    result = [0] * n
    mults = even_mults + odd_mults

    # Standard IFFT formula.
    for k in range(n // 2):
        w = _root_of_unity(n, -k)   # -k since inverse uses the conjugate root of unity.
        product = w * odd[k]        # one complex multiplication per butterfly
        mults += 1

        result[k] = even[k] + product
        result[k + n // 2] = even[k] - product

    return result, mults


def ifft(values: list[int | float | complex]) -> tuple[list[complex], int]:
    """
    Computes the inverse FFT of `values`.
    len(values) must be a power of 2.

    Returns a tuple:
        - result: list of complex numbers, the coefficients of the polynomial
          whose evaluations at the n-th roots of unity are `values`.
        - mults: number of complex multiplications performed during the
          recursive butterfly stage. Note: this does NOT include the n
          divisions used for normalization (x / n), since those are
          divisions, not multiplications — counted separately if you need them.
    """

    n = len(values)
    # Ensure that the length of values is a power of 2.
    if n == 0 or (n & (n - 1) != 0):
        raise ValueError("ifft(), length of values must be a power of 2")

    raw, mults = _ifft_recursive(values)
    # Normalize each coefficient once after recursion.
    result = [x / n for x in raw]
    return result, mults


def fast_poly_multiply(a: list[int] | list[float], b: list[int] | list[float], int_result: bool=True, noise_tolerance: float=1e-10) -> tuple[list[int] | list[float], int]:
    """
    Multiply the polynomials represented by the given lists using the FFT method.

    Parameters:
        a: list of coefficients in order of ascending degree representing the first polynomial.
        b: list of coefficients in order of ascending degree representing the second polynomial.
        int_result: if True, resulting coefficients will be rounded to integers.
        noise_tolerance: float value that represents how close to 0 a number should be to be considered 0.

    Errors if either list is empty.

    Returns a tuple:
        result: the coefficients of the product of a and b in order of ascending degree
        mults: the total number of meaningful multiplication operations performed.
    """

    if (not a) or (not b):
        raise ValueError("multiply_polynomials(), polynomials must not be 0")

    a_padded, b_padded, N = pad_match_lsts(a, b)
   
    a_fft, a_mults = fft(a_padded)
    b_fft, b_mults = fft(b_padded)

    # c_hat computation adds N multiplications.
    c_hat = [a_fft[k] * b_fft[k] for k in range(N)]
    c, c_mults = ifft(c_hat)

    total_mults = a_mults + b_mults + c_mults + N

    return clean_coefficients(c, int_result, noise_tolerance), total_mults


def naive_poly_multiply(a: list[int] | list[float], b: list[int] | list[float], int_coeffs: bool=True, noise_tolerance: float=1e-10) -> tuple[list[int] | list[float], int]:
    """
    Multiply the polynomials represented by the given lists.
    Uses the naive O(n*m) schoolbook convolution.

    Parameters:
        a: list of coefficients in order of ascending degree representing the first polynomial.
        b: list of coefficients in order of ascending degree representing the second polynomial.
        int_result: if True, resulting coefficients will be rounded to integers.
        noise_tolerance: float value that represents how close to 0 a number should be to be considered 0.

    Errors if either list is empty.

    Returns a tuple:
        result: the coefficients of the product of a and b in order of ascending degree
        mults: the total number of multiplication operations performed
    """

    if (not a) or (not b):
        raise ValueError("multiply_polynomials_naive(), polynomials must not be 0")

    n, m = len(a), len(b)
    result = [0] * (n + m - 1)
    mults = 0

    for i in range(n):
        for j in range(m):
            result[i + j] += a[i] * b[j]
            mults += 1

    return clean_coefficients(result, int_coeffs, noise_tolerance), mults


def naive_polymult_count(a: list[int] | list[float], b: list[int] | list[float]) -> int:
    """Returns the number of meaningufl multiplications required to naively multiply polynomials a and b."""
    return len(a)*len(b)