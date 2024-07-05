"""Performance module."""

import numpy as np
import pandas as pd

import performance as pf
import performance3 as pf3
import performance3 as pf2

size = 100_000
n = 100

df = pd.DataFrame(
    {
        "a": np.random.randn(size),
        "b": np.random.randn(size),
        "N": np.random.randint(n, size, (size)),
        "x": "x",
    }
)
print(df.info())

print("Performance 1")
pf.process(df)
print()
print("Performance 2")
pf2.process(df)
print()
print("Performance 3")
pf3.process(df)
