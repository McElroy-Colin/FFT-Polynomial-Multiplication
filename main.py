from FFT import *
from utils import *
from test import *

import json

def main():

    program_mode = input("Choose how to input polynomials, type \"quit\" to quit:\n" \
                        "    <1> for manual coefficient entry\n" \
                        "    <2> for random coefficient generation\n" \
                        "    <3> for read from \"polynomials.txt\" file in CWD\n\n" \
                        ">  ")
    program_mode = clean_input_string(program_mode)

    if program_mode.lower() == "quit":
        quit()

    # Manual entry case
    elif program_mode == "1":
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

    # Random generation case
    elif program_mode == "2":
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
            if degree1 is not False: # Captures both the random degree and entered degree cases.
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

        print(f"\nPolynomial 1:  {coeffs_to_polynomial_string(a)}")
        print(f"Polynomial 2:  {coeffs_to_polynomial_string(b)}\n")

    # Read from file case
    elif program_mode == "3":
        pass # TODO

    else:
        pass # TODO loop all of this

    N = pad_match_polynomials(a, b)

    a_fft = fft(a)
    b_fft = fft(b)

    c_hat = [a_fft[k] * b_fft[k] for k in range(N)]
    c = ifft(c_hat)
    c = clean_coefficients(c)

    print(c)


if __name__ == "__main__":
    main()