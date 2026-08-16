# General utility functions.

import math
import random


def clean_input_string(input_str: str) -> str:
    """
    Cleans the user's input choice by
        - Stripping whitespace
        - Removing any leading '<' and trailing '>' characters
    Returns the cleaned string.
    """
    return input_str.lstrip("< \t").rstrip("> \t")


def parse_coefficients(coeffs_str: str) -> list[int] | list[float] | bool:
    """
    Parses `coeffs_str` as one or more whitespace-separated numeric tokens.

    Returns:
        True          if `coeffs_str` is "quit" (case-insensitive).
        False         if `coeffs_str` is empty/whitespace-only, or any token
                      fails to parse as a finite number.
        list[int]     if every token is a valid integer.
        list[float]   if every token parses, and at least one token
                      is a float (i.e. not every token was an int).
    """

    if coeffs_str.strip().lower() == "quit":
        return True

    # Split on whitespace.
    tokens = coeffs_str.split()

    if not tokens:
        return False

    values = []
    all_ints = True

    for token in tokens:
        try:
            value = float(token)
        except ValueError:
            return False

        if not math.isfinite(value):
            return False

        # Check whether this specific token was written as an int or a float.
        # Note, "4.0" would count as a float, not an int.
        try:
            int(token)
        except ValueError:
            all_ints = False

        values.append(value)

    if all_ints:
        return [int(v) for v in values]
    else:
        return [float(v) for v in values]


def parse_nonneg_int(int_str: str) -> int | bool:
    """
    Parses `int_str` as a non-negative integer (strictly greater than 0).

    Returns:
        True    if `int_str` is "quit" (case-insensitive).
        False   if `int_str` is not a valid integer, or is < 0.
        int     the parsed value, if it is a valid non-negative integer.
    """

    if int_str.strip().lower() == "quit":
        return True

    int_str = int_str.strip()

    try:
        value = int(int_str)
    except ValueError:
        return False

    if value < 0:
        return False

    return value


def _pad_list(lst: list):
    """
    Pads the list `lst` in place with zeros until its length is a power of 2.
    """
    n = 1
    while n < len(lst):
        n *= 2
    lst += [0] * (n - len(lst))


def random_polynomial(
        max_degree: int, 
        always_max_degree: bool=False, 
        coeff_range: tuple[int, int]=(-10, 10), 
        pad_power_of_2: bool=True, 
        int_coeffs: int=True) -> list[int | float]:
    """
    Generates a random list of polynomial coefficients.

    Parameters:
        max_degree: maximums possible degree of the polynomial.
        always_max_degree: if True, the polynomial will always have degree max_degree, otherwise it can be any degree from 0 to max_degree.
        coeff_range: (min, max) tuple describing the allowable range of coefficients.
        pad_power_of_2: if True, the returned list will be padded with zeros to the next power of 2.
        int_coeffs: if True, coefficients will be integers, otherwise they will be floats.

    Errors if max_degree is negative or if coeff_range is invalid.

    Returns the generated list of coefficients, where the i-th element is the coefficient for x^i.
    """

    if max_degree < 0:
        raise ValueError("random_polynomial(), max_degree must be non-negative")
    elif coeff_range[0] > coeff_range[1]:
        raise ValueError("random_polynomial(), coeff_range must be a valid range")

    # Pick a random degree between 0 and max_degree, or always use max_degree if specified.
    degree = max_degree if always_max_degree else random.randint(0, max_degree)

    # Generate random coefficients for the polynomial, including the constant term (degree + 1 total coefficients).
    if int_coeffs:
        coeffs = [random.randint(*coeff_range) for _ in range(degree + 1)]
    else:
        coeffs = [random.uniform(*coeff_range) for _ in range(degree + 1)]

    if pad_power_of_2:
        _pad_list(coeffs)

    return coeffs


def pad_match_polynomials(coeffs1: list[int | float], coeffs2: list[int | float]) -> int:
    """
    Pads two lists of polynomial coefficients in place with zeros so that they have the same length.
    Their lengths will be able to store the product of these polynomials and must be a power of 2.
        i.e. Their lengths will be the next power of 2 greater than or equal to len(coeffs1) + len(coeffs2) - 1.

    Errors if either list is empty.
    
    Returns the new length of the padded lists.
    """

    if (len(coeffs1) == 0) or (len(coeffs2) == 0):
        raise ValueError("pad_match_polynomials(), both lists must be non-empty")

    min_len = len(coeffs1) + len(coeffs2) - 1 
    n = 1
    while n < min_len:
        n *= 2

    coeffs1 += [0] * (n - len(coeffs1))
    coeffs2 += [0] * (n - len(coeffs2))

    return n


def clean_coefficients(coeffs: list[int | float | complex], int_coeffs: bool=True) -> list[int | float]:
    """
    Clean a list of coefficients by 
        - Removing imaginary parts of coefficients
        - Rounding to the nearest integer if int_coeffs is True
        - Removing trailing zeros

    Errors if coeffs is empty.

    Returns the cleaned list of coefficients.
    """

    if len(coeffs) == 0:
        raise ValueError("clean_coefficients(), list of coefficients must be non-empty")

    cleaned = [(round(x.real) if int_coeffs else x.real) for x in coeffs]

    # Remove trailing zeros.
    while cleaned and cleaned[-1] == 0:
        cleaned.pop()

    return cleaned


def coeffs_to_polynomial_string(coeffs: list[int | float]) -> str:
    """
    Converts a list of coefficients to a human-readable polynomial string.
    Errors if coeffs is empty.
    Returns the polynomial string in the form "a_n*x^n + a_(n-1)*x^(n-1) + ... + a_1*x + a_0".
    """

    if len(coeffs) == 0:
        raise ValueError("coeffs_to_polynomial_string(), list of coefficients must be non-empty")

    terms = []
    # Ensure that a ceofficient like "-8" is written as "- 8x^i", not " + -8x^i".
    for i, coeff in enumerate(coeffs):
        if coeff == 0:
            continue
        abs_coeff = -coeff if coeff < 0 else coeff
        term = f"{abs_coeff}" if i == 0 else (f"{abs_coeff}*x^{i}" if abs_coeff != 1 else f"x^{i}")
        sign = "-" if coeff < 0 else "+"
        terms.append((sign, term))

    if not terms:
        return "0"

    first_sign, first_term = terms[0]
    result = f"-{first_term}" if first_sign == "-" else first_term
    for sign, term in terms[1:]:
        result += f" {sign} {term}"

    return result
