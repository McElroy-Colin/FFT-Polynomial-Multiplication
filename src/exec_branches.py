from src.utils import *

import random
import json


def manual_entry_mode() -> tuple[list[int] | list[float], list[int] | list[float]]:
    """Prompt the user to manually enter coefficients for two polynomials."""
    print("Enter the coefficients of the polynomials in ascending order of degree, separated by spaces.")

    while True:
        coeffs_str = input("First polynomial coefficients>  ")
        a = parse_coefficients(coeffs_str)
        if a == True:
            quit()
        elif a is not False:
            coeffs_str = input("Second polynomial coefficients>  ")
            b = parse_coefficients(coeffs_str)
            if b == True:
                quit()
            elif b is not False:
                break

        print("\nCoefficients must be numbers in ascending order of degree, separated by spaces.")

    return a, b


def random_generation_mode() -> tuple[list[int] | list[float], list[int] | list[float]]:
    """Prompt the user for generation settings, then produce two random polynomials."""
    # Read config settings for random generation.
    with open("config.json", "r") as f:
        config = json.load(f)
    MAX_RANDOM_DEGREE = config["max_random_polynomial_degree"]
    COEFFICIENT_RANGE = (config["random_polynomial_min_coeff"], config["random_polynomial_max_coeff"])

    while True:
        int_bool = input("Integer-only coefficients? (Y/n)>  ").strip().lower()

        if int_bool == "quit":
            quit()
        elif int_bool in ["y", "yes"]:
            int_bool = True
            break
        elif int_bool in ["n", "no"]:
            int_bool = False
            break

        print("\nEnter \"Y\"/\"yes\" for integer-only coefficients, \"N\"/\"no\" for float coefficients.")

    print(f"\nEnter the degree of the polynomials to be generated.\n"\
          f"  Leave blank for a random degree between 0 and {MAX_RANDOM_DEGREE}.\n" \
          f"  Polynomials will be generated with random coefficients in the range [{COEFFICIENT_RANGE[0]}, {COEFFICIENT_RANGE[1]}].")

    while True:
        degree_str = input("Degree of the first polynomial>  ")
        degree1 = parse_nonneg_int(degree_str)
        if degree1 == True:
            quit()
        elif degree_str == "":
            degree1 = random.randint(0, MAX_RANDOM_DEGREE)
        if degree1 is not False:  # Captures both the random degree and entered degree cases.
            degree_str = input("Degree of the second polynomial>  ")
            degree2 = parse_nonneg_int(degree_str)
            if degree2 == True:
                quit()
            elif degree_str == "":
                degree2 = random.randint(0, MAX_RANDOM_DEGREE)
                break
            elif degree2:
                break

        print(f"\nDegrees must be non-negative integers, or blank for a random degree between 0 and {MAX_RANDOM_DEGREE}.")

    a = random_polynomial(degree1, always_max_degree=True, coeff_range=COEFFICIENT_RANGE, int_coeffs=int_bool)
    b = random_polynomial(degree2, always_max_degree=True, coeff_range=COEFFICIENT_RANGE, int_coeffs=int_bool)

    # TODO: May not want to print here
    print(f"\nPolynomial 1:  {coeffs_to_polynomial_string(a)}")
    print(f"Polynomial 2:  {coeffs_to_polynomial_string(b)}\n")

    return a, b


def read_from_file_mode() -> tuple[list[int] | list[float], list[int] | list[float]]:
    """Prompt the user for a file path and read polynomial data from it."""
    while True:
        file_path = input("Enter file path, leave blank for default \"input.txt\">  ").strip()
        if file_path == "quit":
            quit()
        elif file_path == "":
            file_path = "input.txt"

        # Ensure the file is readable with a try catch.
        try:
            with open(file_path, "r") as f:
                lines = f.read().splitlines()
        except FileNotFoundError:
            print(f"File \"{file_path}\" not found. Please check the path and try again.\n")
            continue
        except OSError as e:
            print(f"Error reading file \"{file_path}\": {e}\n")
            continue

        lines = [line.strip() for line in lines if line.strip() != ""]  # Technically stripping twice per line here, could be faster

        # TODO: parse `lines` into a and b, then break
        break

    return a, b