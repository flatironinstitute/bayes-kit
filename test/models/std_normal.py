class StdNormal:
    def dims(self):
        return 1
    def log_density(self, params_unc):
        return -0.5 * params_unc[0] * params_unc[0]
        
