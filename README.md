# Polynomial Multiplication: FFT vs. Naive

A small Python project demonstrating the difference between **naive polynomial multiplication** and **FFT-based polynomial multiplication**.

The project is intended to show both an implementation of a fast polynomial multiply using custom implementations of the Fast Fourier Transform 
and Inverse Fourier Transform, as well as to show the operational differences as mentioned above.

## Overview

Given two polynomials,

$$
A(x) = a_0 + a_1x + \dots + a_nx^n
$$

and

$$
B(x) = b_0 + b_1x + \dots + b_mx^m,
$$

their product can be computed in two ways:

* **Naive multiplication** — directly computes every coefficient of the product. This takes approximately **O(nm)** time.
* **FFT multiplication** — uses the Fast Fourier Transform to convert the polynomials into point-value form, multiplies the values pointwise, and transforms the result back. This takes approximately **O(n log n)** time (depending on $\max\{n,m\}$).

This project demonstrates the performance difference as polynomial sizes increase.

## Requirements

* Python 3
* NumPy
* pytest

Install the extra dependencies with:

```bash
pip install -r requirements.txt
```

## Running the Demo

Run the main launcher:

```bash
python main.py
```

The launcher provides a simple way to execute the polynomial multiplication demonstration without needing to invoke individual implementation modules directly.
There is a small `config.json` file with a couple of settings to change and the launcher script provides access to different execution paths.

The `sample_polynomials.txt` file provides $6$ randomly generated polynomials of degree $500$, and it serves as a good demonstration in the different approaches. To
use this file, follow the terminal prompts and make sure to select "human-readable" as the polynomial format.

## Running the Tests

Run the test suite with:

```bash
pytest
```

The tests verify that the FFT-based implementation produces the same polynomial coefficients as the naive implementation for supported inputs.
Note, some tests for the polynomial parser fail around edge cases like double "++" tokens or "3x - 3x" not combining. This can certainly be fixed, but this is not a parsing project so it is not a priority for me.
## How It Works

### Naive Approach

The naive algorithm follows the definition of polynomial multiplication. Every coefficient in the first polynomial is multiplied by every coefficient in the second:

```text
for each coefficient a[i]:
    for each coefficient b[j]:
        result[i + j] += a[i] * b[j]
```

For two polynomials with roughly `n` coefficients each, this requires roughly `n²` multiplications.

### FFT Approach

The FFT approach uses the convolution theorem:

1. Pad both coefficient arrays to an appropriate length.
2. Compute the FFT of each polynomial.
3. Multiply the transformed values element-by-element.
4. Apply the inverse FFT.
5. Round the resulting coefficients to account for floating-point error.

```text
coefficients
     │
     ▼
    FFT
     │
     ▼
point values ── multiply ──► point values
                              │
                              ▼
                            IFFT
                              │
                              ▼
                    product coefficients
```

This reduces the asymptotic complexity from **O(n²)** to approximately **O(n log n)**.

## Correctness

Because NumPy's FFT operates using floating-point arithmetic, the inverse transform may produce very small numerical errors. For example, a coefficient that should mathematically be $10$ might be represented internally as something extremely close to $10$.

The implementation should therefore account for floating-point precision when converting FFT results back into polynomial coefficients.

The test suite compares the two approaches to ensure that their results agree within an appropriate numerical tolerance.

## Complexity

| Approach             | Approximate Complexity |
| -------------------- | ---------------------- |
| Naive multiplication | O(n²)                  |
| FFT multiplication   | O(n log n)             |

The FFT approach has additional setup and numerical overhead, so its theoretical advantage does not necessarily mean it is faster for every input size.
Typically, polynomials of larger degree benefit more from the FFT approach.
