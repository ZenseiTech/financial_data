"""Performance module."""
import numba
from numba import prange

numba.set_num_threads(12)


@numba.jit(fastmath=True, nopython=True)
def f(x):
    """Return the calculation."""
    return x * (x - 1)


@numba.jit(fastmath=True, parallel=True, nopython=True)
def integrate_f(a, b, N):
    """Calculate."""
    s = 0
    dx = (b - a) / N

    for i in prange(N):
        s += f(a + i * dx)

    return s * dx


def process(df):
    """Process start."""
    return df.apply(lambda x: integrate_f(x["a"], x["b"], x["N"]), axis=1)
