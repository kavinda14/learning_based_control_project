import numpy as np


# noinspection PyUnresolvedReferences
def contains(vector, lims):
    return (vector[:, 0] >= lims[:, 0]).all() and (vector[:, 0] <= lims[:, 1]).all()


def sample_vector(lims, damp=0.0):
    # from cube
    dim = lims.shape[0]
    x = np.zeros((dim, 1))
    for i in range(dim):
        x[i] = lims[i, 0] + np.random.uniform(damp, 1 - damp) * (lims[i, 1] - lims[i, 0])
    return x


class Problem:

    def __init__(self):
        self.name = None
        self.num_robots = None
        self.gamma = None

        self.state_dim = None
        self.action_dim = None

        self.policy_encoding_dim = None
        self.value_encoding_dim = None

        self.state_lims = None
        self.action_lims = None
        self.init_lims = None

        self.position_idx = None
        self.state_idxs = None
        self.action_idxs = None

        self.dt = None
        self.times = None

    def sample_action(self):
        return sample_vector(self.action_lims)

    def sample_state(self):
        return sample_vector(self.state_lims)

    def initialize(self):
        valid = False
        state = None
        while not valid:
            state = sample_vector(self.init_lims)
            valid = not self.is_terminal(state)
        return state

    def reward(self, state, action):
        exit("reward needs to be overwritten")

    def normalized_reward(self, state, action):
        exit("normalized_reward needs to be overwritten")

    def step(self, state, action, dt):
        exit("step needs to be overwritten")

    def render(self, states):
        exit("render needs to be overwritten")

    def is_terminal(self, state):
        exit("is_terminal needs to be overwritten")

    def policy_encoding(self, state, robot):
        exit("policy_encoding needs to be overwritten")

    def value_encoding(self, state):
        exit("value_encoding needs to be overwritten")
