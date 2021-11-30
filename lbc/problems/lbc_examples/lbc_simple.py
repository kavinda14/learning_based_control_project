"""
3d double integrator , multi robot uncooperative target
"""
from collections import Counter

import numpy as np
from sympy import sin, cos, atan2, pi

from lbc.problems.problem import Problem, sample_vector
from lbc.reward_functions import prio_reward


def check_agents_param(num_agents: int, orig_parameter):
    # todo  verify working when invalid parameters passed for multi-dimensional parameters (eg location)
    if isinstance(orig_parameter, int):
        orig_parameter = (orig_parameter,)
    if len(orig_parameter) != num_agents:
        each_param = orig_parameter[0]
        orig_parameter = [each_param] * num_agents
    return orig_parameter


def scan(robot_idx, state, scan_radius):
    # todo
    return


class LbcSimple(Problem):

    def __init__(self, num_agents: int = 2, num_regions: int = 8, board_size: int = 10,
                 agent_goals: tuple = ((9, 1), (9, 9)), agent_starts: tuple = ((1, 1), (1, 9)),
                 agent_prios: tuple = (0, 1), agent_speeds: tuple = (1, 1), sensing_ranges: tuple = (4.0, 4.0)):
        """
        State space of individual agent:
            s[0], s[1]:                         location of agent
            s[2]:                               priority of agent
            s[3], s[4]:                         location of goal
            s[5:5+num_regions]:                 distance to closest other agent in a direction
            s[5+num_regions:5+2*num_regions]:   priority of closest other agent in a direction
        Action space of individual agent:
            Single dimension where the value is the agent being able to move in one of a set
            number of directions (default is 8)
        """
        super(LbcSimple, self).__init__()

        self.name = "lbc_simple"
        self.board_size = board_size
        self.num_robots = num_agents
        self.num_regions = num_regions
        self.robot_prios = check_agents_param(num_agents, agent_prios)
        self.sensing_ranges = check_agents_param(num_agents, sensing_ranges)
        self.robot_speeds = check_agents_param(num_agents, agent_speeds)
        # todo  terminal state is when all agents have reached their goal
        self.robot_goals = check_agents_param(num_agents, agent_goals)
        self.robot_start_locs = check_agents_param(num_agents, agent_starts)

        # todo  need to add concept of an obstacle
        #       regions in the space that are not valid locations (need to alter self.isvalid
        self.obstacles = []

        self.dt = 1
        self.gamma = 1.0
        self.prio_bounds = np.asarray([0, 1])
        self.state_dim_per_robot = 21
        self.action_dim_per_robot = 1

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

        self.state_lims = []
        for robot_idx in range(self.num_robots):
            self.state_lims.append([0, self.board_size])
            self.state_lims.append([0, self.board_size])
            self.state_lims.append([np.min(self.prio_bounds), np.max(self.prio_bounds)])
            self.state_lims.append([0, self.board_size])
            self.state_lims.append([0, self.board_size])
            robot_range = self.sensing_ranges[robot_idx]
            # maximum distance any agent is able to sense around it is the max distance for that region
            for _ in range(self.num_regions):
                self.state_lims.append([0, 1])
            for _ in range(self.num_regions):
                self.state_lims.append([np.min(self.prio_bounds), np.max(self.prio_bounds)])

        self.state_lims = np.asarray(self.state_lims).flatten()
        self.state_lims = self.state_lims.reshape(-1, 2)

        # one action to move towards each region + action for remaining still
        self.action_lims = [
            [0, self.num_regions]
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
        # todo generalize starting/goal positions based on number of agents
        #   1: 1/2
        #   2: 1/3, 2/3
        #   3: 1/4, 2/4, 3/4
        #   4: 1/5, 2/5, 3/5, 4/5
        #   ...
        #   ...
        a0_start = self.robot_start_locs[0]
        a0_goal = self.robot_goals[0]
        a0_prio = self.robot_prios[0]

        a1_start = self.robot_start_locs[1]
        a1_goal = self.robot_goals[1]
        a1_prio = self.robot_prios[1]

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
        return start_state.astype('float')

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
        # update robot positions based on action
        for robot_idx in range(self.num_robots):
            # todo  can't move into obstacles
            action = a[robot_idx]
            if action == 0:
                continue
            angle = action * (2 * pi / self.num_regions)
            dx = self.robot_speeds[robot_idx] * cos(angle[0])
            dy = self.robot_speeds[robot_idx] * sin(angle[0])
            s[self.state_idxs[robot_idx][0]] += dx
            s[self.state_idxs[robot_idx][1]] += dy

        # update closest robots in each region
        for robot_idx in range(self.num_robots):
            robot_range = self.sensing_ranges[robot_idx]
            closest_dist = np.full(self.num_regions, robot_range)  # track closest robot in each region
            # loop over all other robots
            for other_robot in [x for x in range(self.num_robots) if x != robot_idx]:
                robot_pos = s[self.state_idxs[robot_idx]][0:2]
                other_robot_pos = s[self.state_idxs[other_robot]][0:2]
                dist_xy = other_robot_pos - robot_pos
                angle = atan2(dist_xy[1], dist_xy[0])
                action_region = round((angle * self.num_regions) / (2 * pi))  # get region where other robot is in
                dist_eucl = np.linalg.norm(dist_xy)

                # todo: edge case - tie-breaking using priorities when 2 agents have same proximity
                # update current other robot to be closest
                if dist_eucl < closest_dist[action_region]:
                    closest_dist[action_region] = dist_eucl
                    # update closest robot distance in state
                    s[self.state_idxs[robot_idx][4 + action_region]] = dist_eucl/robot_range
                    # update priority of closest robot in state
                    s[self.state_idxs[robot_idx][12 + action_region]] = s[self.state_idxs[other_robot][2]]

        return s

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
        # todo  all agents at goal
        all_goal = False
        valid_state = not self.is_valid(state)

        term_criteria = [valid_state, not all_goal]
        return all(term_criteria)

    def is_valid(self, state):
        # todo  clean logic
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
    ########################################
    # Test actions
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
    # Test reward action
    a0_actions = range(1, test_problem.num_regions + 1)
    initial_reward = test_problem.normalized_reward(test_problem.initialize(), None)
    print(f'Reward initial state: {initial_reward[0]}, {initial_reward[1]}')
    for each_action in a0_actions:
        full_action = np.asarray([[0], each_action])
        initial_state = test_problem.initialize()
        next_state = test_problem.step(initial_state, full_action, dt=1)
        next_reward = test_problem.normalized_reward(next_state, full_action)
        print(f'Reward of agent0 taking action {each_action}: {next_reward[0]}')
    ########################################
    # Test scan
    # todo
    ########################################
    # Test step
    # todo
