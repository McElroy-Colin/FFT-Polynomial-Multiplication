from src.utils import *

import random
import json


def manual_entry_mode() -> list[list[int] | list[float]]:
    """Prompt the user to manually enter coefficients for polynomials."""
    print("Enter the coefficients of the polynomials in ascending order of degree, separated by spaces.")

    i = 1
    coeffs_lst = []
    while True:
        coeffs_str = input(f"Polynomial {i} coefficients>  ").strip()
        if coeffs_str == "":
            if len(coeffs_lst) < 2:
                print("Enter at least 2 polynomials to multiply.\n")
                continue
            else:
                break

        curr_poly = parse_coefficients(coeffs_str)
        if curr_poly == True:
            quit()
        elif not curr_poly:
            print("Coefficients must be numbers in ascending order of degree, separated by spaces.\n")
        else:
            coeffs_lst.append(curr_poly)
            i += 1

    return coeffs_lst


def random_generation_mode() -> tuple[list[int] | list[float], list[int] | list[float]]:
    """Prompt the user for generation settings, then produce a specified number of random polynomials."""
    # Read config settings for random generation.
    with open("config.json", "r") as f:
        config = json.load(f)
    MAX_RANDOM_DEGREE = config["max_random_polynomial_degree"]
    COEFFICIENT_RANGE = (config["random_polynomial_min_coeff"], config["random_polynomial_max_coeff"])

    while True:
        int_bool = input("Integer-only coefficients? (Y/n)>  ").strip().lower()

        if int_bool.lower() == "quit":
            quit()
        elif int_bool in ["y", "yes"]:
            int_bool = True
            break
        elif int_bool in ["n", "no"]:
            int_bool = False
            break

        print("\nEnter \"Y\"/\"yes\" for integer-only coefficients, \"N\"/\"no\" for float coefficients.")

    while True:
        num_polys = input("Enter the number of polynomials to multiply>  ").strip().lower()
        num_polys = parse_nonneg_int(num_polys)
        if num_polys == True:
            quit()
        elif (num_polys is not False) and (num_polys > 1):
            break
        
        print(f"Number of polynomials must be an integer greater than 1.\n")

    print(f"\nEnter the degree of the polynomials to be generated.\n"\
          f"    Leave blank for a random degree between 0 and {MAX_RANDOM_DEGREE}.\n" \
          f"    Polynomials will be generated with random coefficients in the range [{COEFFICIENT_RANGE[0]}, {COEFFICIENT_RANGE[1]}].")

    i = 1
    while True:
        degree_str = input(f"Degree of the polynomial >  ")
        max_degree = parse_nonneg_int(degree_str)
        if max_degree == True:
            quit()
        elif degree_str == "":
            max_degree = random.randint(0, MAX_RANDOM_DEGREE)
        if max_degree is not False:  # Captures both the random degree and entered degree cases.
           break

        print(f"\nDegrees must be non-negative integers, or blank for a random degree between 0 and {MAX_RANDOM_DEGREE}.")

    coeffs_lst = []
    for i in range(num_polys):
        coeffs_lst.append(random_polynomial(max_degree, always_max_degree=True, coeff_range=COEFFICIENT_RANGE, int_coeffs=int_bool))
        print(f"Generated polynomial {i + 1}:  {coeffs_to_polynomial_string(coeffs_lst[i])}")

    return coeffs_lst


def read_from_file_mode() -> list[list[int] | list[float]]:
    """Prompt the user for a file path and read polynomial data from it."""
    prompting_file = True
    while prompting_file:
        file_path = input("Enter file path, leave blank for default \"input.txt\">  ").strip()
        if file_path.lower() == "quit":
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

        prompting_file = False

    FORMAT_CHOICES_STR = "\n    <1> for human-readable, e.g. 2x^3 - 5 + 3x^8\n    <2> for ascending degree coefficients, e.g. -2 0 3 16 as -2 + 3x^2 + 16x^3\n>  "
    format_str = input(f"Polynomial format?{FORMAT_CHOICES_STR}").strip()
    while True:
        format_mode = clean_input_string(format_str)
        if format_mode == "1":
            human_readable = True
            break
        elif format_mode == "2":
            human_readable = False
            break
        else:
            format_str = input(f"\nChoice not recognized, please input{FORMAT_CHOICES_STR}").strip()

    lines = [line.strip() for line in lines if line.strip() != ""]
    if human_readable:
        coeffs_lst = [parse_polynomial(line) for line in lines]
    else:
        coeffs_lst = [parse_coefficients(line) for line in lines]

    i = 0
    # Check which lines were properly parsed, skip lines that were not parsed correctly.
    while i < len(coeffs_lst): 
        if human_readable and isinstance(coeffs_lst[i], str):
            print(f"Error parsing polynomial: {lines[i]} due to term \"{coeffs_lst[i]}\". Bypassing...")
            coeffs_lst.pop(i)
        elif (not human_readable) and isinstance(coeffs_lst[i], bool):
            print(f"Error parsing polynomial: {lines[i]}. Bypassing...")
            coeffs_lst.pop(i)
        else:
            i += 1

    return coeffs_lst