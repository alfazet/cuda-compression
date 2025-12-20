'''
A script to test the program's correctness on randomly generated data
'''

import random
import subprocess
import sys


def gen_test(size, rep_chance):
    def random_byte():
        return random.randint(0, 255)

    output = bytearray([0 for _ in range(size)])
    output[0] = random_byte()
    for i in range(1, size):
        if random.random() < rep_chance:
            output[i] = output[i - 1]
        else:
            output[i] = random_byte()
    with open("/tmp/raw", "wb") as f:
        f.write(output)


def run_tests(n_tests, size, rep_chance, method, is_gpu):
    binary = "../build/release/compress"
    device = "gpu" if is_gpu else "cpu"
    print(f"{'*' * 20} {method} [{device}] {'*' * 20}")
    for i in range(n_tests):
        gen_test(size, rep_chance)
        print(f"test {i:04}: ", end="")
        subprocess.call(
            [
                binary,
                "c",
                method,
                "/tmp/raw",
                "/tmp/compressed",
                device,
            ]
        )
        subprocess.call(
            [
                binary,
                "d",
                method,
                "/tmp/compressed",
                "/tmp/decompressed",
                device,
            ]
        )
        if subprocess.call(["diff", "/tmp/raw", "/tmp/decompressed"]) != 0:
            print("FAIL")
            break
        print("OK")


args = sys.argv
if len(args) < 2:
    print(f"usage: {args[0]} binary [n_tests] [size] [rep_chance]")
    exit(1)

binary = args[1]
n_tests = int(args[2]) if len(args) >= 3 else 100
size = int(args[3]) if len(args) >= 4 else (1 << 12)  # number of bytes
rep_chance = float(args[4]) if len(args) >= 5 else 0.1  # chance for a value to repeat
print(
    f"running {n_tests} tests (per method/device) with size = {size}, repetition chance = {rep_chance} ([Enter] to start)"
)
input()
run_tests(n_tests, size, rep_chance, "fl", False)
run_tests(n_tests, size, rep_chance, "fl", True)
