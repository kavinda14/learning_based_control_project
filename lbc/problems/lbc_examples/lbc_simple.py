"""
3d double integrator , multi robot uncooperative target
"""
from collections import Counter

import numpy as np

from lbc.problems.problem import Problem
from lbc.reward_functions import prio_reward


def sample_vector(lims, damp=0.0):
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
        self.sensing_range = 4
        # todo  need to add in linking an agent to a goal position
        #       terminal state is when all agents have reached their goal
        # todo  need to add concept of an obstacle
        #       regions in the space that are not valid locations (need to alter self.isvalid
        self.obstacles = []

        self.state_dim = self.num_robots * self.state_dim_per_robot
        self.action_dim = self.num_robots * self.action_dim_per_robot

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
            self.state_lims.append([0, self.sensing_range])
        for _ in range(self.num_regions):
            self.state_lims.append([np.min(self.prio_bounds), np.max(self.prio_bounds)])
        self.state_lims = self.state_lims * self.num_robots

        self.state_lims = np.asarray(self.state_lims).flatten()
        self.state_lims = self.state_lims.reshape(-1, 2)

        self.action_lims = [
            [1, self.num_regions]
        ]
        self.action_lims = self.action_lims * self.num_robots
        self.action_lims = np.asarray(self.action_lims).flatten()
        self.action_lims = self.action_lims.reshape(-1, 2)
        return

    def sample_action(self):
        return np.rint(sample_vector(self.action_lims))

    def sample_state(self):
        return sample_vector(self.state_lims)

    def initialize(self):
        a0_start = [1, 1]
        a0_goal = [9, 9]
        a0_prio = 0

        a1_start = [1, 9]
        a1_goal = [9, 1]
        a1_prio = 1

        region_dists = [0] * self.num_regions
        region_prios = [0] * self.num_regions
        # These two lines can be used to test these entries in the state spaces as they will
        # create distinct values rather than all 0's
        # region_dists = np.arange(start=0, stop=self.num_regions, step=1)
        # region_prios = np.arange(start=self.num_regions+1, stop=2*self.num_regions+1, step=1)

        start_state = np.hstack((
            a0_start,
            a0_prio,
            a0_goal,
            region_dists,
            region_prios,
            a1_start,
            a1_prio,
            a1_goal,
            region_dists,
            region_prios,
        ))
        return start_state

    def reward(self, state, action):
        s_0 = state[self.state_idxs[0]]
        s_1 = state[self.state_idxs[1]]
        r_0 = prio_reward(s_0, action)
        r_1 = prio_reward(s_1, action)
        rew = np.array([[r_0], [r_1]])
        return rew

    def normalized_reward(self, state, action):
        return self.reward(state, action)

    def step(self, s, a, dt):
        # todo
        # for each robot: slice into state space
        # get angle from action
        # change pos using fixed dist/speed and angle
        # end
        # assign robots their own state spaces
        # for each robot:
        # for all other robots:
        # check if within range, if true and closest, assign to get segment
        import pdb
        pdb.set_trace()
        for robot in range(self.num_robots):
            state = s[self.state_idxs[robot]]
            
        return

    def render(self, states=None, fig=None, ax=None):
        # todo
        if fig is None or ax is None:
            fig, ax = plotter.make_3d_fig()

        if states is not None:

            lims = self.state_lims
            colors = plotter.get_n_colors(self.num_robots)
            for robot in range(self.num_robots):
                robot_state_idxs = self.state_idxs[robot]

                ax.plot(states[:, robot_state_idxs[0]].squeeze(axis=1), states[:, robot_state_idxs[1]].squeeze(axis=1),
                        states[:, robot_state_idxs[2]].squeeze(axis=1), color=colors[robot])
                ax.plot(states[0, robot_state_idxs[0]], states[0, robot_state_idxs[1]], states[0, robot_state_idxs[2]],
                        color=colors[robot], marker='o')
                ax.plot(states[-1, robot_state_idxs[0]], states[-1, robot_state_idxs[1]],
                        states[-1, robot_state_idxs[2]], color=colors[robot], marker='s')

                # projections
                ax.plot(lims[0, 0] * np.ones(states.shape[0]), states[:, robot_state_idxs[1]].squeeze(),
                        states[:, robot_state_idxs[2]].squeeze(),
                        color=colors[robot], linewidth=1, linestyle="--")
                ax.plot(states[:, robot_state_idxs[0]].squeeze(), lims[1, 1] * np.ones(states.shape[0]),
                        states[:, robot_state_idxs[2]].squeeze(),
                        color=colors[robot], linewidth=1, linestyle="--")
                ax.plot(states[:, robot_state_idxs[0]].squeeze(), states[:, robot_state_idxs[1]].squeeze(),
                        lims[2, 0] * np.ones(states.shape[0]),
                        color=colors[robot], linewidth=1, linestyle="--")

            ax.set_xlim((lims[0, 0], lims[0, 1]))
            ax.set_ylim((lims[1, 0], lims[1, 1]))
            ax.set_zlim((lims[2, 0], lims[2, 1]))
            ax.set_box_aspect((lims[0, 1] - lims[0, 0], lims[1, 1] - lims[1, 0], lims[2, 1] - lims[2, 0]))

            for robot in range(self.num_robots):
                ax.scatter(np.nan, np.nan, np.nan, color=colors[robot], label="Robot {}".format(robot))
            ax.legend(loc='best')
        return fig, ax

    def is_terminal(self, state):
        # todo: obstacles
        return not self.is_valid(state)

    def is_valid(self, state):
        # todo: obstacles
        return ((self.state_lims[:, 0] <= state).all() and (state <= self.state_lims[:, 1]).all()).all()

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


if __name__ == '__main__':
    num_actions = 500
    test_problem = LbcSimple()
    action_history = []
    for action_idx in range(0, num_actions):
        each_action = test_problem.sample_action()
        action_history.extend(each_action.flatten().tolist())
    action_counts = Counter(action_history)
    action_counts = sorted(action_counts.items())
    action_history = np.asarray(action_history)
    print(f'min: {np.min(action_history)} | max: {np.max(action_history)}')
    for each_num, each_count in action_counts:
        print(f'\t{each_num}: {each_count}')
    ########################################
    a0_actions = range(1, test_problem.num_regions + 1)
    initial_reward = test_problem.normalized_reward(test_problem.initialize(), None)
    print(f'Reward initial state: {initial_reward[0]}, {initial_reward[1]}')
    for each_action in a0_actions:
        full_action = np.asarray([[0], each_action])
        initial_state = test_problem.initialize()
        next_state = test_problem.step(initial_state, full_action, dt=1)
        next_reward = test_problem.normalized_reward(next_state, full_action)
        print(f'Reward of agent0 taking action {each_action}: {next_reward[0]}')
