from typing import Optional, Union

import numpy as np
from scipy import stats as sst

from bayes_kit.typing import VectorType


class SkewNormal:
    def __init__(
        self,
        a: float = 4,
        loc: Optional[Union[float, VectorType]] = None,
        finite_difference_epsilon: float = 0.000001,
    ) -> None:
        self.a = a
        self._loc = loc if loc is not None else np.array([0])
        self._epsilon = finite_difference_epsilon

    def dims(self) -> int:
        return 1

    def log_density(self, params_unc: VectorType) -> float:
        return sst.skewnorm.logpdf(params_unc, self.a, loc=self._loc)[0]  # type: ignore # scipy is not typed

    def log_density_gradient(self, params_unc: VectorType) -> tuple[float, VectorType]:
        lp = self.log_density(params_unc)
        lp_plus_epsilon = self.log_density(params_unc + self._epsilon)
        return lp, np.array([(lp - lp_plus_epsilon) / self._epsilon])

    def posterior_mean(self) -> float:
        return sst.skewnorm.mean(self.a, loc=self._loc)  # type: ignore # scipy is not typed

    def posterior_variance(self) -> float:
        return sst.skewnorm.var(self.a, loc=self._loc)  # type: ignore # scipy is not typed
