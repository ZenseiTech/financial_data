"""Performance module."""

import time

import numpy as np
import pandas as pd

import performance as pf
import performance3 as pf3
import performance3 as pf2

# import performance_modin as pf5
# import performance_polars as pf4

size = 1_000_000
n = 1000

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


def time_run(name, performance_mod, times=1, is_print_result=True):
    """Time check."""
    print(name)
    start_time = time.time()
    result = run(performance_mod, df, times)
    elapsed_time = time.time() - start_time
    print(f"Current elapsed time: {elapsed_time:.2f} seconds.")
    if is_print_result:
        print(type(result))
    print()


time_run("Performance 1", pf)
time_run("Performance 2", pf2)
time_run("Performance 3", pf3)
# time_run("Performance Modin", pf5)
# time_run("Performance Polars", pf4)
