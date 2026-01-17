"""
A script to run correctness tests for the compression binary. Linux only.
"""

import argparse
import random
import string
import subprocess

TEST_ALPHA = "a"
TEST_RANDOM = "r"
TEST_UNIFORM = "u"


def gen_random_alpha(size):
    return bytes(ord("a") + random.randint(0, 25) for _ in range(size))


def gen_random_random(size):
    return random.randbytes(size)


def gen_random_uniform(size):
    r = random.randint(0, 9)
    return bytes(r for _ in range(size))


def gen_test(path, kind, size):
    with open(path, "wb") as file:
        if kind == TEST_ALPHA:
            file.write(gen_random_alpha(size))
        if kind == TEST_RANDOM:
            file.write(gen_random_random(size))
        if kind == TEST_UNIFORM:
            file.write(gen_random_uniform(size))


def run_tests(binary, n_tests, kind, size, raw, compressed, decompressed, version):
    for method in ["rl"]:
        print(f"\n{method.upper()} TESTS")
        for i in range(0, n_tests):
            gen_test(raw, kind, size)
            print(f"test {i:04}: ", end="")

            res = subprocess.call(
                [
                    binary,
                    "c",
                    method,
                    raw,
                    compressed,
                    version,
                ],
                stdout=subprocess.DEVNULL,
            )
            if res != 0:
                print(f"runtime error while compressing (exit code: {res})")
                return

            res = subprocess.call(
                [
                    binary,
                    "d",
                    method,
                    compressed,
                    decompressed,
                    version,
                ],
                stdout=subprocess.DEVNULL,
            )
            if res != 0:
                print(f"runtime error while decompressing (exit code: {res})")
                return

            if subprocess.call(["cmp", raw, decompressed]) != 0:
                print("fail")
                return
            print("pass")


parser = argparse.ArgumentParser()
parser.add_argument("binary", help="path to the binary file")
parser.add_argument(
    "-n", help="number of tests to run [default: 100]", type=int, default=100
)
parser.add_argument(
    "-t",
    help="type of tests to run ((a)lphabet / (r)andom / (u)niform) [default: (r)]",
    choices=[TEST_ALPHA, TEST_RANDOM, TEST_UNIFORM],
    default=TEST_RANDOM,
)
parser.add_argument(
    "-s",
    help="size of a single test file (in bytes) [default: 2048]",
    type=int,
    default=2048,
)
parser.add_argument("--cpu", help="test the CPU version", action="store_true")
args = parser.parse_args()

binary, n_tests, kind, size, cpu = (
    args.binary,
    args.n,
    args.t,
    args.s,
    args.cpu,
)

subprocess.call("rm -f /tmp/test-*", shell=True)
random_ident = "".join(random.choices(population=string.ascii_lowercase, k=5))
raw_path = f"/tmp/test-{random_ident}-raw"
compressed_path = f"/tmp/test-{random_ident}-compressed"
decompressed_path = f"/tmp/test-{random_ident}-decompressed"
version = "cpu" if cpu else "gpu"

run_tests(
    binary, n_tests, kind, size, raw_path, compressed_path, decompressed_path, version
)
