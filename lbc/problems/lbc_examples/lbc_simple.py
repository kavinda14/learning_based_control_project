"""
3d double integrator , multi robot uncooperative target
"""
import numpy as np

from lbc.problems.problem import Problem, contains
from lbc.reward_functions import prio_reward


def sample_vector(lims, damp=0.0):
    # todo
    # from cube
    dim = lims.shape[0]
    x = np.zeros((dim, 1))
    for i in range(dim):
        x[i] = lims[i, 0] + np.random.uniform(damp, 1 - damp) * (lims[i, 1] - lims[i, 0])
    return x


class LbcSimple(Problem):

    def __init__(self):
        """
        State space of individual agent:
            s[0], s[1]:                         location of agent
            s[2]:                               priority of agent
            s[3], s[4]:                         location of goal
            s[5:5+num_regions]:                 location to closest other agent in a direction
            s[5+num_regions:5+2*num_regions]:   priority of closest other agent in a direction
        Action space of individual agent:
            Single dimension where the value is the agent being able to move in one of a set
            number of directions (default is 8)
        """
        super(LbcSimple, self).__init__()
        self.name = "lbc_simple"
        self.dt = 1
        self.gamma = 1.0

        self.board_size = 10
        self.num_robots = 2
        self.prio_bounds = np.asarray([0, 1])

        self.state_dim_per_robot = 21
        self.action_dim_per_robot = 1
        self.num_regions = 8

        self.state_dim = self.num_robots * self.state_dim_per_robot
        self.action_dim = self.num_robots * self.state_dim_per_robot

        self.policy_encoding_dim = self.state_dim
        self.value_encoding_dim = self.state_dim

        self.state_idxs = [
            each_agent * self.state_dim_per_robot + np.arange(self.state_dim_per_robot)
            for each_agent in range(self.num_robots)
        ]
        self.action_idxs = [
            each_agent * self.action_dim_per_robot + np.arange(self.action_dim_per_robot)
            for each_agent in range(self.num_robots)
        ]

        self.state_lims = [
            [0, self.board_size],
            [0, self.board_size],
            [np.min(self.prio_bounds), np.max(self.prio_bounds)],
            [0, self.board_size],
            [0, self.board_size],
        ]
        for _ in range(self.num_regions):
            self.state_lims.append([0, self.board_size])
        for _ in range(self.num_regions):
            self.state_lims.append([np.min(self.prio_bounds), np.max(self.prio_bounds)])

        self.state_lims = np.asarray(self.state_lims).flatten()
        self.state_lims = self.state_lims.reshape(-1, 2)

        self.action_lims = np.asarray([
            [0, self.num_regions]
        ])
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
        rew = prio_reward(state, action, num_regions=self.num_regions)
        return rew

    def normalized_reward(self, state, action):
        return self.reward(state, action)

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
