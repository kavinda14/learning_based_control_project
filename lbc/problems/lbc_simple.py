"""
3d double integrator , multi robot uncooperative target
"""
import numpy as np

from lbc.problems.problem import Problem, contains


class LbcSimple(Problem):

    def __init__(self):
        super(LbcSimple, self).__init__()
        self.name = "lbcsimple"

        state_dim_per_robot = 6
        action_dim_per_robot = 3

        self.state_idxs = [np.arange(state_dim_per_robot), state_dim_per_robot + np.arange(state_dim_per_robot)]
        self.action_idxs = [np.arange(action_dim_per_robot), action_dim_per_robot + np.arange(action_dim_per_robot)]

        self.num_robots = 2
        self.state_dim = 12
        self.action_dim = 6
        self.policy_encoding_dim = self.state_dim
        self.value_encoding_dim = self.state_dim

        self.desired_distance = 0.5

        self.r_max = 1000
        self.r_min = -1 * self.r_max

        self.t0 = 0
        self.tf = 10
        self.dt = 0.1
        self.gamma = 1.0
        self.mass = 1

        self.position_idx = np.arange(3)
        self.state_control_weight = 1e-5

        self.times = np.arange(self.t0, self.tf, self.dt)

        self.state_lims = None
        self.action_lims = None
        self.init_lims = None

        self.Fc = None
        self.Bc = None
        self.Q = None
        self.Ru = None
        return

    def reward(self, s, a):
        return 0

    def step(self, s, a, dt):
        return

    def render(self, states=None, fig=None, ax=None):
        return fig, ax

    def is_terminal(self, state):
        return not self.is_valid(state)

    def is_valid(self, state):
        return contains(state, self.state_lims)

    def policy_encoding(self, state, robot):
        return state

    def value_encoding(self, state):
        return state
