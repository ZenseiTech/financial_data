"""Polars performance."""
import numba


@numba.jit(fastmath=True)
def f(x):
    """Process."""
    return x * (x - 1)


def integrate_f(a, b, N):
    """Integrate calculation."""
    s = 0
    dx = (b - a) / N

    calcs = []

    index = 0

    for n in N:
        for i in range(n):
            s += f(a[index] + i * dx[index])
        calcs.append(s * dx[index])
        index += 1
        s = 0

    return calcs


def process(df):
    """Process start."""
    return integrate_f(df.get_column("a"), df.get_column("b"), df.get_column("N"))
