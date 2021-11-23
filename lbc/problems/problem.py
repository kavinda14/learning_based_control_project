from lbc.problems import Example0_lbc
from lbc.problems.example1 import Example1
from lbc.problems.example10 import Example10
from lbc.problems.example11 import Example11
from lbc.problems.example12 import Example12
from lbc.problems.example2 import Example2
from lbc.problems.example3 import Example3
from lbc.problems.example4 import Example4
from lbc.problems.example5 import Example5
from lbc.problems.example6 import Example6
from lbc.problems.example7 import Example7
from lbc.problems.example8 import Example8
from lbc.problems.example9 import Example9
from lbc.util import sample_vector


def get_problem_names():
    return Problem.PROBLEM_MAP.keys()


def get_problem(problem_name):
    problem = Problem.PROBLEM_MAP[problem_name.lower()]
    return problem


class Problem:
    PROBLEM_MAP = {
        'example1': Example1,  # 2d single integrator regulator
        'example2': Example2,  # 2d double integrator regulator
        'example3': Example3,  # 3d dubins uncooperative tracking
        'example4': Example4,  # 3d double integrator uncooperative tracking
        'example5': Example5,  # game of atrition
        'example6': Example6,  # bugtrap: 2d single integrator with obstacles
        'example7': Example7,  # 2d double integrator uncooperative tracking
        'example8': Example8,  # 2d single integrator pursuit evasion
        'example9': Example9,  # homicidal chauffeur problem
        'example10': Example10,  # dummy game problem
        'example11': Example11,  # multiscale bugtrap: 2d single integrator with obstacles
        'example12': Example12,  # modified homicidal chauffer
        'example0_lbc': Example0_lbc
    }

    def __init__(self):
        self.num_robots = None
        self.gamma = None
        self.state_dim = None
        self.state_lims = None
        self.init_lims = None
        self.action_dim = None
        self.action_lims = None
        self.position_idx = None
        self.dt = None
        self.times = None
        self.policy_encoding_dim = None
        self.value_encoding_dim = None
        self.name = None

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
