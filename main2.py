"""Performance module."""
import time

# import numpy as np
import pandas as pd
import polars as pl

import performance_numexpr as pf_ne
import performance_original_parallel as pf_original_parallel
import performance_polars as pf_polars

times = 1
print_result = False


def polar_load():
    """Load data for polar."""
    start_time = time.time()
    df = pl.read_csv("output.csv")
    elapsed_time = time.time() - start_time
    filtered_df = df.filter(pl.col("b") > 2)
    print(f"Polar load time: {elapsed_time:.2f} seconds.")
    return filtered_df


def pandas_load():
    """Load data for pandas."""
    start_time = time.time()
    df = pd.read_csv("output.csv", engine="pyarrow")
    elapsed_time = time.time() - start_time
    filtered_df = df[df["b"] > 2]
    print(f"Pandas load time: {elapsed_time:.2f} seconds.")
    return filtered_df


def run(pf, df, times=3):
    """Run generic."""
    for _ in range(times):
        result = pf.process(df)
    return result


def time_polar_run(name, performance_mod, times=1, is_print_result=True):
    """Time check."""
    print(name)
    start_time = time.time()
    df = polar_load()
    print(df.shape)
    result = run(performance_mod, df, times)
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / times
    print(f"Current elapsed time: {avg_time:.2f} seconds.")
    if is_print_result:
        print(result)
    print()


def time_pandas_run(name, performance_mod, times=1, is_print_result=True):
    """Time check."""
    print(name)
    start_time = time.time()
    df = pandas_load()
    print(df.shape)
    result = run(performance_mod, df, times)
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / times
    print(f"Current elapsed time: {avg_time:.2f} seconds.")
    if is_print_result:
        print(result)
    print()


time_polar_run("Performance Polars", pf_polars, times, print_result)

time_pandas_run("Performance Pandas", pf_original_parallel, times, print_result)

time_pandas_run("Performance numexpr", pf_ne, times, print_result)
