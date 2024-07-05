"""Performance module."""
import time

import numba
import numpy as np
import pandas as pd

numba.set_num_threads(12)


@numba.njit(fastmath=True)
def f_plain(x):
    """Return the calculation."""
    return x * (x - 1)


@numba.njit(fastmath=True)
def integrate_f_numba(a, b, N):
    """Return the calc."""
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f_plain(a + i * dx)
    return s * dx


@numba.njit(fastmath=True)
def apply_integrate_f_numba(col_a, col_b, col_N):
    """Return the calc."""
    n = len(col_N)
    result = np.empty(n, dtype="float64")
    assert len(col_a) == len(col_b) == n
    for i in range(n):
        result[i] = integrate_f_numba(col_a[i], col_b[i], col_N[i])
    return result


def compute_numba(df):
    """Return the calc."""
    result = apply_integrate_f_numba(
        df["a"].to_numpy(), df["b"].to_numpy(), df["N"].to_numpy()
    )
    return pd.Series(result, index=df.index, name="result")


def process(df):
    """Process start."""
    start_time = time.time()
    result = compute_numba(df)
    print(result)
    elapsed_time = time.time() - start_time
    print(f"Current elapsed time: {elapsed_time:.2f} seconds.")
