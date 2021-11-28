import abc
from abc import ABC

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


class Problem(ABC):

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

        # position, state, and action idxs are meant as a way to extract information about each individual agent
        # from the full state/action/position vectors
        #   s[state_idxs[0]] is the state vector for the first agent
        #   s[state_idxs[1]] is the state vector for the second agent
        #   ...
        #   ...
        self.position_idx = None
        self.state_idxs = None
        self.action_idxs = None

        self.dt = None
        return

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

    @abc.abstractmethod
    def reward(self, state, action):
        pass

    @abc.abstractmethod
    def normalized_reward(self, state, action):
        pass

    @abc.abstractmethod
    def step(self, state, action, dt):
        pass

    @abc.abstractmethod
    def render(self, states):
        pass

    @abc.abstractmethod
    def is_terminal(self, state):
        pass

    @abc.abstractmethod
    def policy_encoding(self, state, robot):
        pass

    @abc.abstractmethod
    def value_encoding(self, state):
        pass

    @abc.abstractmethod
    def plot_policy_dataset(self, dataset, title, robot):
        pass

    @abc.abstractmethod
    def plot_value_dataset(self, dataset, title):
        pass
