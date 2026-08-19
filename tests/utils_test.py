# Testing functions from utils.py

import math
import pytest

from src.utils import (
    parse_coefficients,
    parse_nonneg_int,
    _pad_list,
    random_polynomial,
    pad_match_lsts,
    clean_coefficients,
    coeffs_to_polynomial_string,
    parse_polynomial,
)

class TestParseCoefficients:
    """Tests for parse_coefficients()."""

    @pytest.mark.parametrize(
        "input_str, expected",
        [
            ("1", [1]),
            ("1 2 3", [1, 2, 3]),
            ("-1 0 5", [-1, 0, 5]),
            ("  1   2   3  ", [1, 2, 3]),
            ("+1 +2 +3", [1, 2, 3]),
            ("-10 20 -30", [-10, 20, -30]),
        ],
    )
    def test_integer_input(self, input_str, expected):
        assert parse_coefficients(input_str) == expected
        assert all(isinstance(x, int) for x in parse_coefficients(input_str))

    @pytest.mark.parametrize(
        "input_str, expected",
        [
            ("1.0", [1.0]),
            ("1.5 2.5 3.5", [1.5, 2.5, 3.5]),
            ("1 2.5 3", [1.0, 2.5, 3.0]),
            ("-1.5 0.25 -3", [-1.5, 0.25, -3.0]),
            ("4.0 5.0", [4.0, 5.0]),
            (".5 1.25", [0.5, 1.25]),
        ],
    )
    def test_float_input(self, input_str, expected):
        result = parse_coefficients(input_str)

        assert result == expected
        assert all(isinstance(x, float) for x in result)

    @pytest.mark.parametrize(
        "input_str",
        [
            "",
            " ",
            "   ",
            "\t",
            "\n",
            " \t\n ",
        ],
    )
    def test_empty_input(self, input_str):
        assert parse_coefficients(input_str) is False

    @pytest.mark.parametrize(
        "input_str",
        [
            "hello",
            "1 hello",
            "1.5 nope",
            "1,2,3",
            "1x",
            "1+2",
        ],
    )
    def test_invalid_input(self, input_str):
        assert parse_coefficients(input_str) is False

    @pytest.mark.parametrize(
        "input_str",
        [
            "nan",
            "NaN",
            "NAN",
            "inf",
            "Inf",
            "infinity",
            "-inf",
            "+inf",
            "-Infinity",
            "+Infinity",
        ],
    )
    def test_non_finite_input(self, input_str):
        assert parse_coefficients(input_str) is False

    @pytest.mark.parametrize(
        "input_str",
        [
            "quit",
            "QUIT",
            "Quit",
            "qUiT",
            " quit ",
            "\tQUIT\t",
        ],
    )
    def test_quit(self, input_str):
        assert parse_coefficients(input_str) is True

    def test_mixed_int_and_float_returns_all_floats(self):
        result = parse_coefficients("1 2.5 3")

        assert result == [1.0, 2.5, 3.0]
        assert all(isinstance(x, float) for x in result)

    def test_integer_looking_float_is_float(self):
        """4.0 is intentionally a float, not an int."""
        result = parse_coefficients("4.0")

        assert result == [4.0]
        assert isinstance(result[0], float)

    def test_scientific_notation(self):
        result = parse_coefficients("1e3 2.5e-1")

        assert result == [1000.0, 0.25]
        assert all(isinstance(x, float) for x in result)


class TestParseNonnegInt:
    """Tests for parse_nonneg_int()."""

    @pytest.mark.parametrize(
        "input_str, expected",
        [
            ("0", 0),
            ("1", 1),
            ("10", 10),
            ("999999", 999999),
            (" 42 ", 42),
            ("\t123\t", 123),
        ],
    )
    def test_valid_integers(self, input_str, expected):
        assert parse_nonneg_int(input_str) == expected

    @pytest.mark.parametrize(
        "input_str",
        [
            "-1",
            "-10",
            "-999",
        ],
    )
    def test_negative_integers(self, input_str):
        assert parse_nonneg_int(input_str) is False

    @pytest.mark.parametrize(
        "input_str",
        [
            "",
            " ",
            "\t",
            "\n",
            "abc",
            "1.5",
            "1.0",
            "1 2",
            "1x",
        ],
    )
    def test_invalid_input(self, input_str):
        assert parse_nonneg_int(input_str) is False

    @pytest.mark.parametrize(
        "input_str",
        [
            "quit",
            "QUIT",
            "Quit",
            " qUiT ",
            "\tQUIT\t",
        ],
    )
    def test_quit(self, input_str):
        assert parse_nonneg_int(input_str) is True

    def test_plus_sign(self):
        assert parse_nonneg_int("+10") == 10


class TestPadList:
    """Tests for _pad_list()."""

    @pytest.mark.parametrize(
        "input_list, expected",
        [
            ([], []),
            ([1], [1]),
            ([1, 2], [1, 2]),
            ([1, 2, 3], [1, 2, 3, 0]),
            ([1, 2, 3, 4], [1, 2, 3, 4]),
            ([1, 2, 3, 4, 5], [1, 2, 3, 4, 5, 0, 0, 0]),
            ([1] * 7, [1] * 7 + [0]),
        ],
    )
    def test_padding(self, input_list, expected):
        original = input_list.copy()

        _pad_list(input_list)

        assert input_list == expected

        # Ensure it mutates the original list rather than returning a new one.
        assert input_list is not original or True

    @pytest.mark.parametrize(
        "n",
        [0, 1, 2, 4, 8, 16, 32, 64],
    )
    def test_power_of_two_lengths_unchanged(self, n):
        values = list(range(n))

        _pad_list(values)

        assert len(values) == n
        assert values == list(range(n))

    def test_mutates_in_place(self):
        values = [1, 2, 3]

        result = _pad_list(values)

        assert result is None
        assert values == [1, 2, 3, 0]


class TestRandomPolynomial:
    """Tests for random_polynomial()."""

    def test_max_degree_zero(self):
        result = random_polynomial(
            max_degree=0,
            always_max_degree=True,
            pad_power_of_2=False,
        )

        assert len(result) == 1

    def test_always_max_degree(self):
        result = random_polynomial(
            max_degree=7,
            always_max_degree=True,
            pad_power_of_2=False,
        )

        assert len(result) == 8

    def test_coefficients_within_range(self):
        result = random_polynomial(
            max_degree=20,
            always_max_degree=True,
            coeff_range=(-5, 5),
            pad_power_of_2=False,
            int_coeffs=True,
        )

        assert all(-5 <= x <= 5 for x in result)

    def test_integer_coefficients(self):
        result = random_polynomial(
            max_degree=10,
            always_max_degree=True,
            pad_power_of_2=False,
            int_coeffs=True,
        )

        assert all(isinstance(x, int) for x in result)

    def test_float_coefficients(self):
        result = random_polynomial(
            max_degree=10,
            always_max_degree=True,
            pad_power_of_2=False,
            int_coeffs=False,
        )

        assert all(isinstance(x, float) for x in result)

    @pytest.mark.parametrize(
        "max_degree",
        [0, 1, 2, 3, 7, 15, 16, 31],
    )
    def test_length_without_padding(self, max_degree):
        result = random_polynomial(
            max_degree=max_degree,
            always_max_degree=True,
            pad_power_of_2=False,
        )

        assert len(result) == max_degree + 1

    @pytest.mark.parametrize(
        "max_degree, expected_length",
        [
            (0, 1),
            (1, 2),
            (2, 4),
            (3, 4),
            (4, 8),
            (7, 8),
            (8, 16),
            (15, 16),
            (16, 32),
        ],
    )
    def test_power_of_two_padding(self, max_degree, expected_length):
        result = random_polynomial(
            max_degree=max_degree,
            always_max_degree=True,
            pad_power_of_2=True,
        )

        assert len(result) == expected_length
        assert len(result) & (len(result) - 1) == 0

    def test_padding_adds_zeros(self):
        result = random_polynomial(
            max_degree=2,
            always_max_degree=True,
            pad_power_of_2=True,
            coeff_range=(1, 1),
        )

        assert result == [1, 1, 1, 0]

    def test_random_degree_is_within_range(self):
        for _ in range(100):
            result = random_polynomial(
                max_degree=10,
                always_max_degree=False,
                pad_power_of_2=False,
            )

            assert 1 <= len(result) <= 11

    @pytest.mark.parametrize(
        "max_degree",
        [-1, -2, -100],
    )
    def test_negative_max_degree(self, max_degree):
        with pytest.raises(
            ValueError,
            match="max_degree must be non-negative",
        ):
            random_polynomial(max_degree)

    def test_invalid_coefficient_range(self):
        with pytest.raises(
            ValueError,
            match="coeff_range must be a valid range",
        ):
            random_polynomial(
                max_degree=5,
                coeff_range=(10, -10),
            )

    def test_single_value_coefficient_range(self):
        result = random_polynomial(
            max_degree=5,
            always_max_degree=True,
            coeff_range=(7, 7),
            pad_power_of_2=False,
        )

        assert result == [7, 7, 7, 7, 7, 7]


class TestPadMatchLists:
    """Tests for pad_match_lsts()."""

    @pytest.mark.parametrize(
        "a, b, expected_a, expected_b, expected_n",
        [
            (
                [1],
                [2],
                [1],
                [2],
                1,
            ),
            (
                [1, 2],
                [3],
                [1, 2],
                [3, 0],
                2,
            ),
            (
                [1, 2, 3],
                [4],
                [1, 2, 3, 0],
                [4, 0, 0, 0],
                4,
            ),
            (
                [1, 2, 3],
                [4, 5],
                [1, 2, 3, 0],
                [4, 5, 0, 0],
                4,
            ),
            (
                [1, 2, 3, 4, 5],
                [6, 7],
                [1, 2, 3, 4, 5, 0, 0, 0],
                [6, 7, 0, 0, 0, 0, 0, 0],
                8,
            ),
        ],
    )
    def test_padding(
        self,
        a,
        b,
        expected_a,
        expected_b,
        expected_n,
    ):
        result_a, result_b, n = pad_match_lsts(a, b)

        assert result_a == expected_a
        assert result_b == expected_b
        assert n == expected_n

    def test_does_not_modify_inputs(self):
        a = [1, 2, 3]
        b = [4, 5]

        original_a = a.copy()
        original_b = b.copy()

        pad_match_lsts(a, b)

        assert a == original_a
        assert b == original_b

    @pytest.mark.parametrize(
        "a, b",
        [
            ([], [1]),
            ([1], []),
            ([], []),
        ],
    )
    def test_empty_lists(self, a, b):
        with pytest.raises(
            ValueError,
            match="both lists must be non-empty",
        ):
            pad_match_lsts(a, b)

    @pytest.mark.parametrize(
        "a, b",
        [
            ([1], [2]),
            ([1, 2], [3]),
            ([1, 2, 3], [4, 5]),
            ([1, 2, 3, 4, 5], [6, 7, 8]),
        ],
    )
    def test_result_length_is_power_of_two(self, a, b):
        _, _, n = pad_match_lsts(a, b)

        assert n & (n - 1) == 0
        assert n >= len(a) + len(b) - 1


class TestCleanCoefficients:
    """Tests for clean_coefficients()."""

    def test_empty_list_raises(self):
        with pytest.raises(
            ValueError,
            match="list of coefficients must be non-empty",
        ):
            clean_coefficients([], True, 1e-10)

    def test_removes_imaginary_part(self):
        result = clean_coefficients(
            [1 + 2j, 3 - 4j],
            int_coeffs=False,
            noise_tolerance=1e-10,
        )

        assert result == [1, 3]

    def test_rounds_integer_coefficients(self):
        result = clean_coefficients(
            [1.2, 2.7, -3.4, -4.8],
            int_coeffs=True,
            noise_tolerance=1e-10,
        )

        assert result == [1, 3, -3, -5]

    def test_preserves_float_coefficients(self):
        result = clean_coefficients(
            [1.2, 2.7, -3.4],
            int_coeffs=False,
            noise_tolerance=1e-10,
        )

        assert result == [1.2, 2.7, -3.4]

    @pytest.mark.parametrize(
        "value",
        [
            0.0,
            1e-12,
            -1e-12,
            1e-15,
            -1e-15,
        ],
    )
    def test_removes_numerical_noise(self, value):
        result = clean_coefficients(
            [1.0, value],
            int_coeffs=False,
            noise_tolerance=1e-10,
        )

        assert result == [1.0]

    def test_preserves_values_above_noise_tolerance(self):
        result = clean_coefficients(
            [1e-5],
            int_coeffs=False,
            noise_tolerance=1e-10,
        )

        assert result == [1e-5]

    def test_removes_trailing_zeros(self):
        result = clean_coefficients(
            [1, 2, 3, 0, 0],
            int_coeffs=True,
            noise_tolerance=1e-10,
        )

        assert result == [1, 2, 3]

    def test_preserves_zero_polynomial(self):
        result = clean_coefficients(
            [0, 0, 0],
            int_coeffs=True,
            noise_tolerance=1e-10,
        )

        assert result == [0]

    def test_preserves_internal_zeros(self):
        result = clean_coefficients(
            [1, 0, 3, 0],
            int_coeffs=True,
            noise_tolerance=1e-10,
        )

        assert result == [1, 0, 3]

    def test_complex_values_with_small_imaginary_component(self):
        result = clean_coefficients(
            [1 + 1e-15j, 2 - 1e-15j],
            int_coeffs=False,
            noise_tolerance=1e-10,
        )

        assert result == [1.0, 2.0]

    def test_integer_fft_output(self):
        result = clean_coefficients(
            [1.00000000001, 2.99999999999, -4.00000000001],
            int_coeffs=True,
            noise_tolerance=1e-10,
        )

        assert result == [1, 3, -4]


class TestCoeffsToPolynomialString:
    """Tests for coeffs_to_polynomial_string()."""

    @pytest.mark.parametrize(
        "coeffs, expected",
        [
            ([0], "0"),
            ([1], "1"),
            ([-1], "-1"),
            ([5], "5"),
            ([-5], "-5"),
            ([1, 2], "1 + 2*x^1"),
            ([1, -2], "1 - 2*x^1"),
            ([3, 2, 1], "3 + 2*x^1 + x^2"),
            ([-3, -2, -1], "-3 - 2*x^1 - x^2"),
            ([1, 0, 3], "1 + 3*x^2"),
            ([0, 1], "x^1"),
            ([0, -1], "-x^1"),
            ([0, 0, 1], "x^2"),
            ([5, 0, 0, -2], "5 - 2*x^3"),
            ([0, 0, 0], "0"),
        ],
    )
    def test_conversion(self, coeffs, expected):
        assert coeffs_to_polynomial_string(coeffs) == expected

    def test_float_coefficients(self):
        result = coeffs_to_polynomial_string([1.5, -2.25, 3.75])

        assert result == "1.5 - 2.25*x^1 + 3.75*x^2"

    def test_float_one_is_not_printed_as_coefficient(self):
        result = coeffs_to_polynomial_string([0, 1.0, -1.0])

        assert result == "x^1 - x^2"

    def test_skips_zero_coefficients(self):
        result = coeffs_to_polynomial_string([1, 0, 0, 4])

        assert result == "1 + 4*x^3"

    def test_empty_list_raises(self):
        with pytest.raises(
            ValueError,
            match="list of coefficients must be non-empty",
        ):
            coeffs_to_polynomial_string([])

    def test_negative_first_coefficient(self):
        assert coeffs_to_polynomial_string([-3, 2]) == "-3 + 2*x^1"

    def test_negative_nonconstant_coefficient(self):
        assert coeffs_to_polynomial_string([3, -2]) == "3 - 2*x^1"


class TestParsePolynomial:
    """Tests for parse_polynomial()."""

    @pytest.mark.parametrize(
        "poly, expected",
        [
            ("3x^2 + 2x - 5", [-5, 2, 3]),
            ("-x + 2.5x^2 + 3", [3.0, -1.0, 2.5]),
            ("-x^3 + 4", [4, 0, 0, -1]),
            ("x", [0, 1]),
            ("-x", [0, -1]),
            ("x^2", [0, 0, 1]),
            ("-x^2", [0, 0, -1]),
            ("5", [5]),
            ("-5", [-5]),
            ("0", [0]),
        ],
    )
    def test_basic_polynomials(self, poly, expected):
        assert parse_polynomial(poly) == expected

    @pytest.mark.parametrize(
        "poly",
        [
            "3x^2+2x-5",
            "3x^2 + 2x - 5",
            "3x^2\t+\t2x\t-\t5",
            "  3x^2   +   2x   -   5  ",
        ],
    )
    def test_whitespace(self, poly):
        assert parse_polynomial(poly) == [-5, 2, 3]

    @pytest.mark.parametrize(
        "poly, expected",
        [
            ("1.5", [1.5]),
            ("1.5x", [0.0, 1.5]),
            ("2.5x^2 + 1", [1.0, 0.0, 2.5]),
            ("x + 2.5x^2", [0.0, 1.0, 2.5]),
            ("1 + 2x + 3.5x^2", [1.0, 2.0, 3.5]),
        ],
    )
    def test_decimal_coefficients(self, poly, expected):
        result = parse_polynomial(poly)

        assert result == expected
        assert all(isinstance(x, float) for x in result)

    def test_decimal_anywhere_makes_all_coefficients_float(self):
        result = parse_polynomial("1 + 2x + 3.5x^2")

        assert result == [1.0, 2.0, 3.5]
        assert all(type(x) is float for x in result)

    @pytest.mark.parametrize(
        "poly, expected",
        [
            ("x", [0, 1]),
            ("-x", [0, -1]),
            ("+x", [0, 1]),
            ("3x", [0, 3]),
            ("-3x", [0, -3]),
            ("3.5x", [0.0, 3.5]),
        ],
    )
    def test_first_degree_terms(self, poly, expected):
        assert parse_polynomial(poly) == expected

    @pytest.mark.parametrize(
        "poly, expected",
        [
            ("x^0", [1]),
            ("5x^0", [5]),
            ("-3x^0", [-3]),
            ("2.5x^0", [2.5]),
        ],
    )
    def test_zero_degree_terms(self, poly, expected):
        assert parse_polynomial(poly) == expected

    @pytest.mark.parametrize(
        "poly, expected",
        [
            ("x^3", [0, 0, 0, 1]),
            ("2x^5", [0, 0, 0, 0, 0, 2]),
            ("-x^4", [0, 0, 0, 0, -1]),
        ],
    )
    def test_missing_degrees_are_zero(self, poly, expected):
        assert parse_polynomial(poly) == expected

    def test_terms_can_be_out_of_order(self):
        result = parse_polynomial("2x^3 + 4 + 3x")

        assert result == [4, 3, 0, 2]

    def test_like_terms_are_combined(self):
        result = parse_polynomial("2x + 3x")

        assert result == [0, 5]

    def test_like_terms_can_cancel(self):
        result = parse_polynomial("3x - 3x")

        assert result == [0]

    def test_constant_terms_are_combined(self):
        result = parse_polynomial("2 + 3")

        assert result == [5]

    @pytest.mark.parametrize(
        "poly",
        [
            "abc",
            "x^",
            "^2",
            "x^^2",
            "3xx",
            "3x^2x",
            "2**x",
            "x^2.5",
            "x^-2",
            "1,2",
        ],
    )
    def test_invalid_terms(self, poly):
        result = parse_polynomial(poly)

        assert isinstance(result, str)

    def test_invalid_term_is_returned(self):
        result = parse_polynomial("3x^2 + garbage + 1")

        assert result == "garbage"

    @pytest.mark.parametrize(
        "poly",
        [
            "x^2 +",
            "+ x",
            "3x^2 ++ 2x",
            "--x",
        ],
    )
    def test_malformed_polynomials(self, poly):
        result = parse_polynomial(poly)

        assert isinstance(result, str)

    def test_negative_terms(self):
        result = parse_polynomial("-3x^3 - 2x^2 - x - 5")

        assert result == [-5, -1, -2, -3]

    def test_positive_and_negative_terms(self):
        result = parse_polynomial("3x^3 - 2x^2 + x - 5")

        assert result == [-5, 1, -2, 3]

    def test_all_zero_coefficients(self):
        result = parse_polynomial("0x^3 + 0x^2 + 0x + 0")

        assert result == [0, 0, 0, 0]

    def test_large_exponent(self):
        result = parse_polynomial("2x^10")

        assert len(result) == 11
        assert result[10] == 2
        assert all(x == 0 for x in result[:10])

    def test_decimal_constant_with_integer_terms(self):
        result = parse_polynomial("2x^2 + 3x + 4.5")

        assert result == [4.5, 3.0, 2.0]
        assert all(isinstance(x, float) for x in result)


