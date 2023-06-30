import numpy as np
import numpy.typing as npt


class StdNormal:
    def dims(self) -> int:
        return 1

    def log_density(self, params_unc: npt.NDArray[np.float64]) -> float:
        theta: float = params_unc[0]
        return -0.5 * theta * theta

    def log_density_gradient(
        self, params_unc: npt.NDArray[np.float64]
    ) -> tuple[float, npt.NDArray[np.float64]]:
        return -0.5 * params_unc[0] * params_unc[0], -params_unc

    def posterior_mean(self) -> float:
        return 0

    def posterior_variance(self) -> float:
        return 1
