"""Performance module."""
import numba
from numba import prange

numba.set_num_threads(12)


@numba.jit(fastmath=True)
def f(x):
    """Return the calculation."""
    return x * (x - 1)


@numba.jit(fastmath=True, parallel=True)
def integrate_f(a, b, N):
    """Calculate."""
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
    return integrate_f(df["a"].to_numpy(), df["b"].to_numpy(), df["N"].to_numpy())
