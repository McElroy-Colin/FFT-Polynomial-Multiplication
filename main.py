from src.polymath import *
from src.utils import *
from src.exec_branches import *


def main():
    CHOICES_STR = "\n    <1> for manual coefficient entry\n" \
                    "    <2> for random coefficient generation\n" \
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

    if not coeffs_lst:
        print(f"Failed to parse at least 2 polynomials. Check formatting and try again...")
        quit()
        
    # Multiply each polynomial iteratively using the FFT and IFFT.
    a = coeffs_lst[0]
    for i in range(1, len(coeffs_lst)):
        b = coeffs_lst[i]
        a = multiply_polynomials(a, b)

    result = a
    readable_result = coeffs_to_polynomial_string(result)

    


if __name__ == "__main__":
    main()