# Soft launcher script

import time
import json

from src.polymath import *
from src.utils import *
from src.exec_branches import *


def main():
    CHOICES_STR = "\n    <1> for manual entry\n" \
                    "    <2> for random generation\n" \
                    "    <3> to read from a file\n\n" \
                  ">  "

    program_mode = input(f"Choose how to input polynomials, type \"quit\" to quit:{CHOICES_STR}")

    while True:
        program_mode = clean_input_string(program_mode)

        if program_mode.lower() == "quit":
            quit()

        # Manual entry case
        elif program_mode == "1":
            coeffs_lst = manual_entry_mode()
            break

        # Random generation case
        elif program_mode == "2":
            coeffs_lst = random_generation_mode()
            break

        # Read from file case
        elif program_mode == "3":
            coeffs_lst = read_from_file_mode()
            break

        # No choice case
        else:
            program_mode = input(f"\nChoice not recognized, please input{CHOICES_STR}")

    if len(coeffs_lst) < 2:
        print(f"Failed to parse at least 2 polynomials. Check formatting and try again...\n")
        quit()

    max_degree = max(len(coeffs) for coeffs in coeffs_lst) - 1
    min_degree = min(len(coeffs) for coeffs in coeffs_lst) - 1
        
    total_fft_mults = 0
    total_naive_mults = 0

    output_str = f"\nMultiplying\n  ({coeffs_to_polynomial_string(coeffs_lst[0])})\n"

    for i in range(1, len(coeffs_lst)):
        output_str += f"* ({coeffs_to_polynomial_string(coeffs_lst[i])})\n"

    start = time.perf_counter()

    # Multiply each polynomial iteratively using the FFT and IFFT.
    a = coeffs_lst[0]
    for i in range(1, len(coeffs_lst)):
        int_result = True
        b = coeffs_lst[i]

        # Check the final coefficient of each since it is guaranteed to not be 0.
        if isinstance(a[len(a) - 1], float) or isinstance(b[len(b) - 1], float):
            int_result = False

        # Count multiplications for the naive approach.
        total_naive_mults += naive_polymult_count(a, b)

        # Multiply and count multiplications using the FFT/IFFT method.
        a, fft_mults = fast_poly_multiply(a, b, int_result)
        total_fft_mults += fft_mults
    result = a

    end = time.perf_counter()

    readable_result = coeffs_to_polynomial_string(result)

    # Retrieve output path from config
    with open("config.json", "r") as f:
        config = json.load(f)
    OUTPUT_FILE = config["output_file_path"]

    # Print and store the first half of output.
    print(output_str)
    with open(OUTPUT_FILE, "w") as out:
        out.write(output_str)

    # Formulate the second half of output.
    output_str = f"\nProduct:\n{readable_result}\n\n"
    if max_degree == min_degree:
        output_str += f"Succesfully multiplied {len(coeffs_lst)} polynomials of degree {max_degree} (product above)\n"
    else:
        output_str += f"Succesfully multiplied {len(coeffs_lst)} polynomials between degrees {min_degree} and {max_degree} (product above)\n"

    output_str += \
        f"Product degree:  {len(result) - 1}\n" \
        f"Multiplications for naive approach:  {total_naive_mults}\nMultiplications for FFT approach:  {total_fft_mults}\n" \
        f"FFT approach used {(min(total_naive_mults, total_fft_mults) / max(total_naive_mults, total_fft_mults) * 100):.2f}% of multiplications required for the naive approach.\n" \
        f"Elapsed time:  {end - start:.6f} s"
    
    print(output_str)
    with open(OUTPUT_FILE, "a") as out:
        out.write(output_str)


if __name__ == "__main__":
    main()
