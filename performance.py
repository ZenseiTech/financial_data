"""Performance module."""
import time

import numba

numba.set_num_threads(12)


@numba.jit(fastmath=True)
def f(x):
    """Return the calculation."""
    return x * (x - 1)


@numba.jit(fastmath=True)
def integrate_f(a, b, N):
    """Calculate."""
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
    start_time = time.time()
    result = integrate_f(df["a"].to_numpy(), df["b"].to_numpy(), df["N"].to_numpy())
    elapsed_time = time.time() - start_time
    print(f"Current elapsed time: {elapsed_time:.2f} seconds.")
    print(result)
