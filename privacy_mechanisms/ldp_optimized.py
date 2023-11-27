import numpy as np
import cvxpy as cp

# class Pufferfish_Binary_Attr:
#     def __init__(self, joint_prob, array = ["0 0", "0 1", "1 0", "1 1"]):
#         self.ARRAY = array
#         self.joint_prob = np.array(joint_prob)
#         if (abs(np.sum(self.joint_prob) - 1) > 0.00001):
#             raise Exception("Probability distribution must sum to 1: sum is %.4f" % np.sum(self.joint_prob))
 
#     def __str__(self):
#         return f"Joint probability = {self.joint_prob}"
    
#     def cal_probabilities(self, actual_value, eps):
#         prob_vec = self.joint_prob/np.exp(eps)
#         prob_vec[actual_value] += 1 - 1/np.exp(eps)

#         return prob_vec

#     def gen_random_output(self, actual_value, eps, out_index_):
#         prob_vec = self.cal_probabilities(actual_value, eps)
#         random_ = np.random.choice(len(self.ARRAY) if out_index_ else self.ARRAY, 1, p=prob_vec)
#         return random_

class LDP_mechanism:
    def __init__(self, joint_prob, array = ["0 0", "0 1", "1 0", "1 1"]):
        self.ARRAY = array
        self.joint_prob = np.array(joint_prob)
        # self.alpha = np.min(self.joint_prob)
        if (abs(np.sum(self.joint_prob) - 1) > 0.00001):
            raise Exception("Probability distribution must sum to 1: sum is %.4f" % np.sum(self.joint_prob))
 
    def __str__(self):
        return f"Joint probability = {self.joint_prob}"
    
    def cal_probabilities(self, eps, alpha = 0.1, n = 4, err_ord = 2):
        prob_matrix = self.get_optimize(self.joint_prob, eps, alpha, n, err_ord) # alpha - kl distance, n - no_states
        return prob_matrix

    def gen_random_output(self, actual_value, out_index_, prob_matrix):
        prob_vec = prob_matrix[actual_value]
        random_ = np.random.choice(len(self.ARRAY) if out_index_ else self.ARRAY, 1, p=prob_vec)
        return random_
    
    def get_optimize(self, Z, EPS, Alpha, n, err_ord):
        '''
            Optimization
        '''

        Z = [0.25, 0.25, 0.25, 0.25]

        index_dict = {"1": [0, 0], "2": [0, 1], "3": [1, 0], "4": [1, 1]}

        def point_wise_err(value_tuple):
            if err_ord == -1:
                return 1 if abs(value_tuple[0] - value_tuple[1]) > 0 else 0
            else:
                return np.linalg.norm(np.array(index_dict[str(value_tuple[0])])-np.array(index_dict[str(value_tuple[1])]), ord=err_ord)

        # Parameters
        # n = 4  # Dimensions of the matrix
        # Alpha = 0.01 # KL divergence upper bound
        # EPS = 1 # Privacy budget
        n = int(n)
        print("n ", n, " alpha ", Alpha, " EPS ", EPS)
        U = np.zeros((n, n))

        for i in range(n):
            for j in range(n):
                U[i][j] = point_wise_err([j+1, i+1])
                
        print(U)

        # Define the optimization variable
        X = cp.Variable((n, n)) # P

        # Objective function
        objective = cp.Minimize(np.transpose(Z)@cp.sum(cp.multiply(X, U), axis=1)) #cp.Minimize((X@np.transpose(U))@Z)

        # Constraints
        X_T = X.T
        Y = X_T@Z
        constraints = [cp.sum(X, axis=1) == 1,  # Each row sums to 1
                    X >= 0.000001] #, # Non-negative elements
                    # cp.sum(cp.kl_div(Y, Z)) <= Alpha]  # KL divergence

        # Adding KL divergence constraints for each row
        for i in range(n):
            for j in range(n):
                for k in range(n):
                    if j == k:
                        continue
                    constraints.append(X[j, i] - np.exp(EPS)*X[k, i] <= 0)
                    # print(f"X[{j}, {i}] - np.exp(EPS)*X[{k}, {i}] <= 0")

        # Define and solve the problem
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS, verbose=False)

        # Output the optimized matrix
        matrix = np.maximum(np.array(X.value), 0)
        row_sums = matrix.sum(axis=1, keepdims=True)
        print("Optimized Matrix P:\n", X.value)
        print("Original Distribution Z:\n", Z)
        print("Perturbed Distribution Z:\n", np.matmul(np.transpose(Z),(matrix/row_sums)))
        return matrix/row_sums

def example():
    p_m = Pufferfish_Binary_Attr([0.5, 0, 0, 0.5])
    print(p_m.cal_probabilities(1, 1))
    for i in range(400):
        print(p_m.gen_random_output(1, 1))

    print(str(p_m))

