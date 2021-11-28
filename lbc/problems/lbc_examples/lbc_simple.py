"""
3d double integrator , multi robot uncooperative target
"""
import numpy as np

from lbc.problems.problem import Problem, contains


def sample_vector(lims, damp=0.0):
    # from cube
    dim = lims.shape[0]
    x = np.zeros((dim, 1))
    for i in range(dim):
        x[i] = lims[i, 0] + np.random.uniform(damp, 1 - damp) * (lims[i, 1] - lims[i, 0])
    return x


class LbcSimple(Problem):

    def __init__(self):
        super(LbcSimple, self).__init__()
        self.name = "lbcsimple"
        self.dt = 0.1
        self.gamma = 1.0

        self.num_robots = 2
        self.state_dim = 12
        self.action_dim = 6
        self.policy_encoding_dim = self.state_dim
        self.value_encoding_dim = self.state_dim

        state_dim_per_robot = 6
        action_dim_per_robot = 3

        self.state_idxs = [np.arange(state_dim_per_robot), state_dim_per_robot + np.arange(state_dim_per_robot)]
        self.action_idxs = [np.arange(action_dim_per_robot), action_dim_per_robot + np.arange(action_dim_per_robot)]

        self.state_lims = None
        self.action_lims = None
        return

    def sample_action(self):
        return sample_vector(self.action_lims)

    def sample_state(self):
        return sample_vector(self.state_lims)

    def initialize(self):
        # todo
        # valid = False
        # state = None
        # while not valid:
        #     state = sample_vector(self.init_lims)
        #     valid = not self.is_terminal(state)
        return sample_vector(self.state_lims)

    def reward(self, state, action):
        # todo
        return 0

    def normalized_reward(self, state, action):
        # todo
        return 0

    def step(self, s, a, dt):
        # todo
        return

    def render(self, states=None, fig=None, ax=None):
        # todo
        return fig, ax

    def is_terminal(self, state):
        return not self.is_valid(state)

    def is_valid(self, state):
        return contains(state, self.state_lims)

    def policy_encoding(self, state, robot):
        return state

    def value_encoding(self, state):
        return state

    def plot_policy_dataset(self, dataset, title, robot):
        # todo
        pass

    def plot_value_dataset(self, dataset, title):
        # todo
        pass
