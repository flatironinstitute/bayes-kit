import numpy.typing as npt
import numpy as np
from scipy import stats as sst
from typing import Union
from numpy.typing import NDArray
from numpy import float64


class SkewNormal:
    def __init__(
        self,
        a: float = 4,
        loc: Union[float, NDArray[float64]] = 0
    ) -> None:
        self.a = a
        self.loc = loc

    def dims(self) -> int:
        return 1

    def log_density(self, params_unc: npt.NDArray[np.float64]) -> float:
        return sst.skewnorm.logpdf(params_unc, self.a, loc=self.loc)[0]

    def log_density_gradient(
        self, params_unc: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64]]:
        raise NotImplementedError

    def posterior_mean(self) -> float:
        mean: Union[float, NDArray[float64]] = sst.skewnorm.mean(self.a, loc=self.loc) # type: ignore # scipy is not typed
        if isinstance(mean, float):
            return mean
        return mean[0]

    def posterior_variance(self) -> float:
        var: Union[float, NDArray[float64]] = sst.skewnorm.var(self.a, loc=self.loc) # type: ignore # scipy is not typed
        if isinstance(var, float):
            return var
        return var[0]
