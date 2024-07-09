"""Original improvement from Pandas website.

https://pandas.pydata.org/docs/user_guide/enhancingperf.html.
"""
import numba
import numpy as np
import pandas as pd


@numba.jit
def f_plain(x):
    """Plain function."""
    return x * (x - 1)


@numba.jit
def integrate_f_numba(a, b, N):
    """Integrate numba."""
    s = 0
    dx = (b - a) / N
    for i in range(N):
        s += f_plain(a + i * dx)
    return s * dx


@numba.jit
def apply_integrate_f_numba(col_a, col_b, col_N):
    """Apply integrate numba."""
    n = len(col_N)
    result = np.empty(n, dtype="float64")
    assert len(col_a) == len(col_b) == n
    for i in range(n):
        result[i] = integrate_f_numba(col_a[i], col_b[i], col_N[i])
    return result


def process(df):
    """Process."""
    result = apply_integrate_f_numba(
        df["a"].to_numpy(), df["b"].to_numpy(), df["N"].to_numpy()
    )
    return pd.Series(result, index=df.index, name="result")
