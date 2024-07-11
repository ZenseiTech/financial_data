"""Polars performance."""
import numba
from numba import prange

numba.set_num_threads(12)


@numba.jit(fastmath=True, nopython=True)
def f(x):
    """Process."""
    return x * (x - 1)


@numba.jit(fastmath=True, parallel=True, nopython=True)
def integrate_f(a, b, N):
    """Integrate calculation."""
    s = 0
    dx = (b - a) / N

    calcs = []

    index = 0

    for n in N:
        for i in prange(n):
            s += f(a[index] + i * dx[index])
        calcs.append(s * dx[index])
        index += 1
        s = 0

    return calcs


def process(df):
    """Process start."""
    return integrate_f(
        df.get_column("a").to_numpy(),
        df.get_column("b").to_numpy(),
        df.get_column("N").to_numpy(),
    )
