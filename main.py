"""Performance module."""

import time

import numpy as np
import pandas as pd

import performance as pf
import performance3 as pf3
import performance3 as pf2

# import performance_polars as pf4

size = 100_000
n = 100

df = pd.DataFrame(
    {
        "a": np.random.randn(size),
        "b": np.random.randn(size),
        "N": np.random.randint(n, size, (size)),
        "x": "x",
    }
)
print(df.info())
print()


def run(pf, df, times=1):
    """Run generic."""
    for i in range(times):
        result = pf.process(df)
    return result


times = 3

print("Performance 1")
start_time = time.time()
result = run(pf, df, times)
elapsed_time = time.time() - start_time
print(f"Current elapsed time: {elapsed_time:.2f} seconds.")
# print(result)
print()

print("Performance 2")
start_time = time.time()
result = run(pf2, df, times)
elapsed_time = time.time() - start_time
print(f"Current elapsed time: {elapsed_time:.2f} seconds.")
# print(result)
print()

print("Performance 3")
start_time = time.time()
result = run(pf3, df, times)
elapsed_time = time.time() - start_time
print(f"Current elapsed time: {elapsed_time:.2f} seconds.")
# print(result)
print()

# print("Performance Polars")
# start_time = time.time()
# df = pl.from_pandas(df)
# result = run(pf4, df, times)
# elapsed_time = time.time() - start_time
# print(f"Current elapsed time: {elapsed_time:.2f} seconds.")
# # print(result)
# print()
