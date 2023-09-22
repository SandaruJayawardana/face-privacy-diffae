import numpy as np
from privacy_mechanism import *

class Randomized_Response_Binary_Attr():
    
    def __init__(self, joint_prob, array = ["0 0", "0 1", "1 0", "1 1"]):
        self.ARRAY = array
        self.joint_prob = np.array(joint_prob)
        if (abs(np.sum(self.joint_prob) - 1) > 0.00001):
            raise Exception("Probability distribution must sum to 1: sum is %.4f" % np.sum(self.joint_prob))
 
    def __str__(self):
        return f"Joint probability = {self.joint_prob}"
    
    def cal_probabilities(self, actual_value, eps):
        prob_vec = np.ones(4)/(np.exp(eps) + 3)
        prob_vec[actual_value] *= np.exp(eps)

        return prob_vec

    def gen_random_output(self, actual_value, eps, out_index_):
        prob_vec = self.cal_probabilities(actual_value, eps)
        random_ = np.random.choice(len(self.ARRAY) if out_index_ else self.ARRAY, 1, p=prob_vec)
        return random_
