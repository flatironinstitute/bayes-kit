from bayes_kit.typing import VectorType


class StdNormal:
    def dims(self) -> int:
        return 1

    def log_density(self, params_unc: VectorType) -> float:
        theta: float = params_unc[0]
        return -0.5 * theta * theta

    def log_density_gradient(self, params_unc: VectorType) -> tuple[float, VectorType]:
        return -0.5 * params_unc[0] * params_unc[0], -params_unc

    def posterior_mean(self) -> float:
        return 0

    def posterior_variance(self) -> float:
        return 1
