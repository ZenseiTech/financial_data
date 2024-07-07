"""Performance module."""
# import modin.pandas as pd


def f(x):
    """Return the calculation."""
    return x * (x - 1)


def integrate_f(a, b, N):
    """Calculate."""
    s = 0
    dx = (b - a) / N

    for i in range(N):
        s += f(a + i * dx)

    return s * dx


def process(df):
    """Process start."""
    return df.apply(lambda x: integrate_f(x["a"], x["b"], x["N"]), axis=1)
