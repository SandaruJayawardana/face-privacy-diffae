import numpy as np

class laplace_mechanism:
    def __init__(self, sensitivity):
        self.sensitivity = sensitivity
 
    def __str__(self):
        return f"Lapalce, sensitivity = {self.sensitivity}"
    
    def cal_probabilities(self, actual_value, eps, sens = -1):
        raise("Laplace mechanism does not have this method")

    def gen_random_output(self, actual_value, eps, sens = -1):
        if sens == -1:
            sens = self.sensitivity
        prob_vec = np.random.laplace(0, sens/eps)

        return actual_value+prob_vec
    
def local_sensitivity(data, clip = 0.1):
    min_ = np.min(data)
    max_ = np.max(data)

    return abs(max_ - min_) * (1 - clip)
