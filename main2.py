"""Performance module."""
import time

# import numpy as np
import pandas as pd
import polars as pl

import performance_original_parallel as pf_original_parallel
import performance_polars as pf_polars

times = 1
print_result = False


def run(pf, df, times=3):
    """Run generic."""
    for _ in range(times):
        result = pf.process(df)
    return result


def time_polar_run(name, performance_mod, times=1, is_print_result=True):
    """Time check."""
    print(name)
    start_time = time.time()
    df = pl.read_csv("output.csv")
    filtered_df = df.filter(pl.col("b") > 2)
    print(filtered_df.shape)
    result = run(performance_mod, filtered_df, times)
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
    df = pd.read_csv("output.csv", engine="pyarrow")
    # df = pd.read_csv("output.csv")
    filtered_df = df[df["b"] > 2]
    print(filtered_df.shape)
    result = run(performance_mod, filtered_df, times)
    elapsed_time = time.time() - start_time
    avg_time = elapsed_time / times
    print(f"Current elapsed time: {avg_time:.2f} seconds.")
    if is_print_result:
        print(result)
    print()


time_polar_run("Performance Polars", pf_polars, times, print_result)

time_pandas_run("Performance Pandas", pf_original_parallel, times, print_result)
