class StdNormal:
    def dims(self):
        return 1
    def log_density(self, params_unc):
        return -0.5 * params_unc[0] * params_unc[0]

    def log_density_gradient(self, params_unc):
        return -0.5 * params_unc[0] * params_unc[0], -params_unc

    def posterior_mean(self):
        return 0

    def posterior_variance(self):
        return 1
