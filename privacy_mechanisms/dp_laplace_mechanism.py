import numpy as np

class laplace_mechanism:
    def __init__(self, sensitivity):
        self.sensitivity = sensitivity
 
    def __str__(self):
        return f"Lapalce, sensitivity = {self.sensitivity}"
    
    def cal_probabilities(self, actual_value, eps, sens = -1):
        raise("Laplace mechanism does not have this method")

    def gen_random_output(self, actual_value, eps, sens = -1, min_ = 0, max_ = 0):
        if sens == -1:
            sens = self.sensitivity
        prob_vec = np.random.laplace(0, sens/eps)

        perturbed_value = actual_value+prob_vec

        if perturbed_value > max_:
            perturbed_value = max_
        elif perturbed_value < min_:
            perturbed_value = min_
        
        return perturbed_value
    
def local_sensitivity_and_min_max(data, clip = 0.1):
    min_ = np.min(data)
    max_ = np.max(data)

    sensitivity = abs(max_ - min_)
    min_clipped = min_ + sensitivity * clip
    max_clipped = max_ - sensitivity * clip

    return sensitivity, min_clipped, max_clipped
