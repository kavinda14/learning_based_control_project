import numpy as np

from lbc.solvers.solver import Solver


class Empty(Solver):

    def __init__(self):
        super().__init__()
        self.solver_name = "empty"

    def policy(self, problem, state):
        action_dim = problem.action_dim
        return np.zeros((action_dim, 1))
