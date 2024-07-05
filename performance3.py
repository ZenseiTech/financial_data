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

    # calcs = []

    for i in range(N):
        # calcs.append(a + i * dx)
        s += f(a + i * dx)

    return s * dx


def process(df):
    """Process start."""
    start_time = time.time()
    result = df.apply(lambda x: integrate_f(x["a"], x["b"], x["N"]), axis=1)
    elapsed_time = time.time() - start_time
    print(f"Current elapsed time: {elapsed_time:.2f} seconds.")
    print(result)
