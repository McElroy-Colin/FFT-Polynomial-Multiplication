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
            a, b = manual_entry_mode()
            break

        # Random generation case
        elif program_mode == "2":
            a, b = random_generation_mode()
            break

        # Read from file case
        elif program_mode == "3":
            a, b = read_from_file_mode()
            break

        # No choice case
        else:
            program_mode = input(f"\nChoice not recognized, please input{CHOICES_STR}")

    # TODO: The rest of this should be in the src file for it...

    N = pad_match_polynomials(a, b)

    a_fft = fft(a)
    b_fft = fft(b)

    c_hat = [a_fft[k] * b_fft[k] for k in range(N)]
    c = ifft(c_hat)
    c = clean_coefficients(c)

    print(c)


if __name__ == "__main__":
    main()