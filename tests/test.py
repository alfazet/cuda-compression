import os
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
n_tests = int(args[1]) if len(args) >= 2 else 100
size = int(args[2]) if len(args) >= 3 else (1 << 12)  # number of bytes
rep_chance = float(args[3]) if len(args) >= 4 else 0.1  # chance for a value to repeat

print(f"running tests with size = {size}, repetition chance = {rep_chance}")
run_tests(n_tests, size, rep_chance, "fl", False)
run_tests(n_tests, size, rep_chance, "fl", True)
