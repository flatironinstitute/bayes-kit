import numpy as np
from typing import Union
from numpy.typing import NDArray, ArrayLike


ArrayType = NDArray[np.float64]
SeedType = Union[int, np.random.BitGenerator, np.random.Generator]


__all__ = [
    "ArrayLike",
    "ArrayType",
    "SeedType",
]
