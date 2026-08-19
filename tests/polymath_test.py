import numpy as np
import pytest


from src.polymath import fft, ifft, fast_poly_multiply, naive_poly_multiply, naive_polymult_count
from src.utils import random_polynomial


class TestFFT:
    """Tests for the forward FFT implementation."""

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128])
    def test_against_numpy(self, n):
        """FFT should agree with numpy.fft.fft."""
        coeffs = random_polynomial(
            max_degree=n - 1,
            always_max_degree=True,
            coeff_range=(-10, 10),
            pad_power_of_2=True,
            int_coeffs=True,
        )

        result, mults = fft(coeffs)
        expected = np.fft.fft(np.asarray(coeffs, dtype=complex))

        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

        # One multiplication per butterfly, with
        # n/2 butterflies per recursion level.
        expected_mults = (n // 2) * (n.bit_length() - 1)
        assert mults == expected_mults

    @pytest.mark.parametrize("n", [2, 4, 8, 16, 32])
    def test_random_float_polynomials(self, n):
        """FFT should work correctly with floating-point coefficients."""
        coeffs = random_polynomial(
            max_degree=n - 1,
            always_max_degree=True,
            coeff_range=(-10, 10),
            pad_power_of_2=True,
            int_coeffs=False,
        )

        result, _ = fft(coeffs)
        expected = np.fft.fft(np.asarray(coeffs, dtype=complex))

        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    @pytest.mark.parametrize("n", [2, 4, 8, 16, 32])
    def test_random_complex_polynomials(self, n):
        """FFT should work correctly with complex coefficients."""
        rng = np.random.default_rng(12345)
        coeffs = (
            rng.uniform(-10, 10, n)
            + 1j * rng.uniform(-10, 10, n)
        ).tolist()

        result, _ = fft(coeffs)
        expected = np.fft.fft(np.asarray(coeffs, dtype=complex))

        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_single_coefficient(self):
        """FFT of one coefficient should return the coefficient unchanged."""
        result, mults = fft([7])

        assert result == [7]
        assert mults == 0

    def test_zero_polynomial(self):
        """FFT of the zero polynomial should be all zeros."""
        coeffs = [0] * 16

        result, mults = fft(coeffs)

        np.testing.assert_allclose(result, np.zeros(16))
        assert mults == 32

    @pytest.mark.parametrize("coeffs", [
        [],
        [1, 2, 3],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6, 7],
        [1] * 6,
        [1] * 10,
    ])
    def test_non_power_of_two_lengths(self, coeffs):
        """FFT should reject lengths that are not powers of two."""
        with pytest.raises(ValueError, match="length of coeffs must be a power of 2"):
            fft(coeffs)

    @pytest.mark.parametrize("n", [2, 4, 8, 16, 32, 64])
    def test_linearity(self, n):
        """FFT(a + b) = FFT(a) + FFT(b)."""
        rng = np.random.default_rng(12345)

        a = rng.uniform(-10, 10, n)
        b = rng.uniform(-10, 10, n)

        result_a, _ = fft(a.tolist())
        result_b, _ = fft(b.tolist())
        result_sum, _ = fft((a + b).tolist())

        np.testing.assert_allclose(
            np.asarray(result_sum),
            np.asarray(result_a) + np.asarray(result_b),
            rtol=1e-12,
            atol=1e-12,
        )


class TestIFFT:
    """Tests for the inverse FFT implementation."""

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32, 64, 128])
    def test_against_numpy(self, n):
        """IFFT should agree with numpy.fft.ifft."""
        rng = np.random.default_rng(12345)

        values = (
            rng.uniform(-10, 10, n)
            + 1j * rng.uniform(-10, 10, n)
        ).tolist()

        result, mults = ifft(values)
        expected = np.fft.ifft(np.asarray(values, dtype=complex))

        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

        expected_mults = (n // 2) * (n.bit_length() - 1)
        assert mults == expected_mults

    @pytest.mark.parametrize("n", [2, 4, 8, 16, 32])
    def test_random_polynomials_round_trip(self, n):
        """IFFT(FFT(coeffs)) should recover the original coefficients."""
        coeffs = random_polynomial(
            max_degree=n - 1,
            always_max_degree=True,
            coeff_range=(-10, 10),
            pad_power_of_2=True,
            int_coeffs=True,
        )

        values, fft_mults = fft(coeffs)
        result, ifft_mults = ifft(values)

        np.testing.assert_allclose(
            result,
            np.asarray(coeffs, dtype=complex),
            rtol=1e-12,
            atol=1e-12,
        )

        expected_mults = (n // 2) * (n.bit_length() - 1)
        assert fft_mults == expected_mults
        assert ifft_mults == expected_mults

    @pytest.mark.parametrize("n", [2, 4, 8, 16, 32])
    def test_random_float_values(self, n):
        """IFFT should correctly invert arbitrary complex frequency values."""
        rng = np.random.default_rng(54321)

        values = (
            rng.uniform(-10, 10, n)
            + 1j * rng.uniform(-10, 10, n)
        ).tolist()

        result, _ = ifft(values)
        expected = np.fft.ifft(np.asarray(values, dtype=complex))

        np.testing.assert_allclose(result, expected, rtol=1e-12, atol=1e-12)

    def test_single_value(self):
        """IFFT of one value should return that value."""
        result, mults = ifft([7])

        assert result == [7]
        assert mults == 0

    def test_zero_values(self):
        """IFFT of all zeros should return all zeros."""
        values = [0] * 16

        result, mults = ifft(values)

        np.testing.assert_allclose(result, np.zeros(16))
        assert mults == 32

    @pytest.mark.parametrize("values", [
        [],
        [1, 2, 3],
        [1, 2, 3, 4, 5],
        [1, 2, 3, 4, 5, 6, 7],
        [1] * 6,
        [1] * 10,
    ])
    def test_non_power_of_two_lengths(self, values):
        """IFFT should reject lengths that are not powers of two."""
        with pytest.raises(ValueError, match="length of values must be a power of 2"):
            ifft(values)

    @pytest.mark.parametrize("n", [2, 4, 8, 16, 32, 64])
    def test_matches_numpy_after_forward_fft(self, n):
        """
        FFT followed by our IFFT should match both the original input
        and NumPy's IFFT.
        """
        rng = np.random.default_rng(98765)

        coeffs = (
            rng.uniform(-10, 10, n)
            + 1j * rng.uniform(-10, 10, n)
        ).tolist()

        values, _ = fft(coeffs)
        result, _ = ifft(values)

        numpy_result = np.fft.ifft(np.fft.fft(coeffs))

        np.testing.assert_allclose(
            result,
            numpy_result,
            rtol=1e-12,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            result,
            np.asarray(coeffs),
            rtol=1e-12,
            atol=1e-12,
        )

    @pytest.mark.parametrize("n", [2, 4, 8, 16, 32])
    def test_round_trip_random_polynomial(self, n):
        """
        Test the intended polynomial interpretation:
        evaluating coefficients with FFT and recovering them with IFFT.
        """
        coeffs = random_polynomial(
            max_degree=n - 1,
            always_max_degree=False,
            coeff_range=(-100, 100),
            pad_power_of_2=True,
            int_coeffs=True,
        )

        # random_polynomial() may produce a length smaller than n when
        # max_degree is not always used, so derive the actual size.
        n_actual = len(coeffs)

        values, _ = fft(coeffs)
        recovered, _ = ifft(values)

        np.testing.assert_allclose(
            recovered,
            np.asarray(coeffs, dtype=complex),
            rtol=1e-12,
            atol=1e-12,
        )


class TestFastPolyMultiply:
    """Tests for fast_poly_multiply()."""

    @pytest.mark.parametrize(
        "a, b",
        [
            ([1], [2]),
            ([1, 2], [3, 4]),
            ([1, 2, 3], [4, 5]),
            ([1, -2, 3], [-2, 4]),
            ([0, 1], [1, 2, 3]),
            ([1, 0, 0, 2], [3, 4]),
        ],
    )
    def test_known_values(self, a, b):
        """Fast multiplication should produce known correct results."""
        result, _ = fast_poly_multiply(a, b)

        expected = np.polynomial.polynomial.polymul(a, b)

        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 10, 16, 20, 32])
    def test_against_numpy(self, n):
        """Fast polynomial multiplication should agree with NumPy."""
        a = random_polynomial(
            max_degree=n - 1,
            always_max_degree=True,
            coeff_range=(-10, 10),
            pad_power_of_2=False,
            int_coeffs=True,
        )
        b = random_polynomial(
            max_degree=n - 1,
            always_max_degree=True,
            coeff_range=(-10, 10),
            pad_power_of_2=False,
            int_coeffs=True,
        )

        result, _ = fast_poly_multiply(a, b, True)
        expected = np.polynomial.polynomial.polymul(a, b)

        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32])
    def test_random_float_polynomials(self, n):
        """Fast multiplication should work with floating-point coefficients."""
        a = random_polynomial(
            max_degree=n - 1,
            always_max_degree=True,
            coeff_range=(-10, 10),
            pad_power_of_2=False,
            int_coeffs=False,
        )
        b = random_polynomial(
            max_degree=n - 1,
            always_max_degree=True,
            coeff_range=(-10, 10),
            pad_power_of_2=False,
            int_coeffs=False,
        )

        result, _ = fast_poly_multiply(a, b, False)
        expected = np.polynomial.polynomial.polymul(a, b)

        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-10,
            atol=1e-10,
        )

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32])
    def test_matches_naive(self, n):
        """Fast multiplication and naive multiplication should agree."""
        a = random_polynomial(
            max_degree=n - 1,
            always_max_degree=True,
            coeff_range=(-10, 10),
            pad_power_of_2=False,
            int_coeffs=True,
        )
        b = random_polynomial(
            max_degree=n - 1,
            always_max_degree=True,
            coeff_range=(-10, 10),
            pad_power_of_2=False,
            int_coeffs=True,
        )

        fast_result, _ = fast_poly_multiply(a, b, True)
        naive_result, _ = naive_poly_multiply(a, b, True)

        np.testing.assert_allclose(
            fast_result,
            naive_result,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_constant_polynomials(self):
        """Multiplication of constant polynomials."""
        result, _ = fast_poly_multiply([5], [7])

        assert result == [35]

    def test_zero_polynomial(self):
        """Multiplication by the zero polynomial should produce zero."""
        result, _ = fast_poly_multiply([0], [1, 2, 3])

        assert result == [0]

    @pytest.mark.parametrize(
        "a, b",
        [
            ([], [1]),
            ([1], []),
            ([], []),
        ],
    )
    def test_empty_polynomial(self, a, b):
        """Empty polynomials should raise ValueError."""
        with pytest.raises(
            ValueError,
            match="polynomials must not be 0",
        ):
            fast_poly_multiply(a, b)

    @pytest.mark.parametrize(
        "a, b",
        [
            ([1, 2, 3], [4, 5]),
            ([1, 0, 2], [3, 4, 5]),
            ([0, 1], [0, 0, 3]),
        ],
    )
    def test_multiplication_count(self, a, b):
        """Fast multiplication should report its expected multiplication count."""
        _, mults = fast_poly_multiply(a, b)

        # pad_match_lsts determines the FFT size N.
        max_len = max(len(a), len(b))
        product_len = len(a) + len(b) - 1
        N = 1

        while N < product_len:
            N *= 2

        # Two forward FFTs + one inverse FFT.
        fft_mults = 3 * (N // 2) * int(np.log2(N))

        # N pointwise multiplications.
        expected = fft_mults + N

        assert mults == expected


class TestNaivePolyMultiply:
    """Tests for naive_poly_multiply()."""

    @pytest.mark.parametrize(
        "a, b",
        [
            ([1], [2]),
            ([1, 2], [3, 4]),
            ([1, 2, 3], [4, 5]),
            ([1, -2, 3], [-2, 4]),
            ([0, 1], [1, 2, 3]),
            ([1, 0, 0, 2], [3, 4]),
        ],
    )
    def test_known_values(self, a, b):
        """Naive multiplication should produce known correct results."""
        result, _ = naive_poly_multiply(a, b)

        expected = np.polynomial.polynomial.polymul(a, b)

        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 5, 8, 10, 16, 20, 32])
    def test_against_numpy(self, n):
        """Naive polynomial multiplication should agree with NumPy."""
        a = random_polynomial(
            max_degree=n - 1,
            always_max_degree=True,
            coeff_range=(-10, 10),
            pad_power_of_2=False,
            int_coeffs=True,
        )
        b = random_polynomial(
            max_degree=n - 1,
            always_max_degree=True,
            coeff_range=(-10, 10),
            pad_power_of_2=False,
            int_coeffs=True,
        )

        result, _ = naive_poly_multiply(a, b, True)
        expected = np.polynomial.polynomial.polymul(a, b)

        np.testing.assert_array_equal(result, expected)

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32])
    def test_random_float_polynomials(self, n):
        """Naive multiplication should work with floating-point coefficients."""
        a = random_polynomial(
            max_degree=n - 1,
            always_max_degree=True,
            coeff_range=(-10, 10),
            pad_power_of_2=False,
            int_coeffs=False,
        )
        b = random_polynomial(
            max_degree=n - 1,
            always_max_degree=True,
            coeff_range=(-10, 10),
            pad_power_of_2=False,
            int_coeffs=False,
        )

        result, _ = naive_poly_multiply(a, b, False)
        expected = np.polynomial.polynomial.polymul(a, b)

        np.testing.assert_allclose(
            result,
            expected,
            rtol=1e-12,
            atol=1e-12,
        )

    @pytest.mark.parametrize("n", [1, 2, 4, 8, 16, 32])
    def test_matches_fast(self, n):
        """Naive multiplication and fast multiplication should agree."""
        a = random_polynomial(
            max_degree=n - 1,
            always_max_degree=True,
            coeff_range=(-10, 10),
            pad_power_of_2=False,
            int_coeffs=True,
        )
        b = random_polynomial(
            max_degree=n - 1,
            always_max_degree=True,
            coeff_range=(-10, 10),
            pad_power_of_2=False,
            int_coeffs=True,
        )

        naive_result, _ = naive_poly_multiply(a, b, True)
        fast_result, _ = fast_poly_multiply(a, b, True)

        np.testing.assert_allclose(
            naive_result,
            fast_result,
            rtol=1e-10,
            atol=1e-10,
        )

    def test_constant_polynomials(self):
        """Multiplication of constant polynomials."""
        result, mults = naive_poly_multiply([5], [7])

        assert result == [35]
        assert mults == 1

    def test_zero_polynomial(self):
        """Multiplication by the zero polynomial should produce zero."""
        result, mults = naive_poly_multiply([0], [1, 2, 3])

        assert result == [0]
        assert mults == 3

    @pytest.mark.parametrize(
        "a, b",
        [
            ([], [1]),
            ([1], []),
            ([], []),
        ],
    )
    def test_empty_polynomial(self, a, b):
        """Empty polynomials should raise ValueError."""
        with pytest.raises(
            ValueError,
            match="polynomials must not be 0",
        ):
            naive_poly_multiply(a, b)

    @pytest.mark.parametrize(
        "a, b",
        [
            ([1], [2]),
            ([1, 2], [3, 4]),
            ([1, 2, 3], [4, 5]),
            ([1, 2, 3, 4], [5, 6, 7]),
        ],
    )
    def test_multiplication_count(self, a, b):
        """Naive multiplication should perform exactly n*m multiplications."""
        _, mults = naive_poly_multiply(a, b)

        assert mults == len(a) * len(b)


class TestNaivePolymultCount:
    """Tests for naive_polymult_count()."""

    @pytest.mark.parametrize(
        "a, b, expected",
        [
            ([1], [2], 1),
            ([1, 2], [3], 2),
            ([1, 2], [3, 4], 4),
            ([1, 2, 3], [4, 5], 6),
            ([1, 2, 3], [4, 5, 6], 9),
            ([1] * 10, [1] * 20, 200),
        ],
    )
    def test_count(self, a, b, expected):
        """The multiplication count should be len(a) * len(b)."""
        assert naive_polymult_count(a, b) == expected

    @pytest.mark.parametrize("n", [1, 2, 3, 4, 8, 16, 32, 100])
    def test_square_count(self, n):
        """Multiplying two polynomials of the same size requires n^2 multiplications."""
        a = [1] * n
        b = [2] * n

        assert naive_polymult_count(a, b) == n * n

    def test_does_not_depend_on_coefficients(self):
        """The count depends only on the input lengths."""
        assert naive_polymult_count(
            [100, -50, 0, 17],
            [3, 0, -8],
        ) == 12