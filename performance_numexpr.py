"""Performance module."""
import numba
import numexpr as ne
import numpy as np
import pandas as pd
from numba import prange

numba.set_num_threads(12)


@numba.jit(fastmath=True, nopython=True)
def f(x):
    """Return the calculation."""
    return x * (x - 1)


@numba.jit(fastmath=True, parallel=True, nopython=True)
def integrate_f(a, N, dx):
    """Calculate."""
    s = 0

    calcs = []

    index = 0

    for n in N:
        for i in prange(n):
            s += f(a[index] + i * dx[index])
        calcs.append(s * dx[index])
        index += 1
        s = 0

    return np.array(calcs)


def process(df):
    """Process start."""
    dx = ne.evaluate("(b - a) / N", local_dict=df)
    result = integrate_f(df["a"].to_numpy(), df["N"].to_numpy(), dx)
    return pd.Series(result)
