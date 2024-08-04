"""Performance module."""

import time

import numpy as np
import pandas as pd
import polars as pl

import performance_apply as pf_apply
import performance_numexpr as pf_ne

# import performance_original as pf_original
import performance_original_parallel as pf_original_parallel
import performance_polars as pf_polars
import performance_vectorized as pf_vectorized

# import performance_modin as pf5


size = 1_000_000
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


def run(pf, df, times=3):
    """Run generic."""
    for _ in range(times):
        result = pf.process(df)
    return result


def time_run(name, performance_mod, times=1, is_print_result=True):
    """Time check."""
    print(name)
    start_time = time.time()
    result = run(performance_mod, df, times)
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / times
    print(f"Current elapsed time: {avg_time:.2f} seconds.")
    if is_print_result:
        print(result)
    print()


times = 1
print_result = False
# time_run("Performance original", pf_original, times, print_result)
time_run("Performance original parallel", pf_original_parallel, times, print_result)
time_run("Performance vectorized", pf_vectorized, times, print_result)
time_run("Performance apply", pf_apply, times, print_result)
time_run("Performance numexpr", pf_ne, times, print_result)

# time_run("Performance Modin", pf5)

df = pl.from_pandas(df)
time_run("Performance Polars", pf_polars, times, print_result)
