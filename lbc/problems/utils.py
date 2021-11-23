"""
@title
@description
"""

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
from lbc.solvers.empty import Empty
from lbc.solvers.mcts import MCTS
from lbc.solvers.policy_solver import PolicySolver
from lbc.solvers.puct_v1 import PUCT_V1
from lbc.solvers.puct_v2 import PUCT_V2

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
}


def get_problem_names():
    return PROBLEM_MAP.keys()


def get_problem(problem_name):
    problem = PROBLEM_MAP[problem_name.lower()]
    return problem


def get_solver(solver_name, policy_oracle=None, value_oracle=None, search_depth=10, number_simulations=1000,
               C_pw=2.0, alpha_pw=0.5, C_exp=1.0, alpha_exp=0.25, beta_policy=0.0, beta_value=1.0, vis_on=False):
    if solver_name == "Empty":
        solver = Empty()
    elif solver_name == "MCTS":
        solver = MCTS()
    elif solver_name == "PUCT_V1":
        solver = PUCT_V1(
            policy_oracle=policy_oracle,
            value_oracle=value_oracle,
            search_depth=search_depth,
            number_simulations=number_simulations,
            C_pw=C_pw,
            alpha_pw=alpha_pw,
            C_exp=C_exp,
            alpha_exp=alpha_exp,
            beta_policy=beta_policy,
            beta_value=beta_value,
            vis_on=vis_on
        )

    elif solver_name == "PUCT_V2":
        solver = PUCT_V2(
            policy_oracle=policy_oracle,
            value_oracle=value_oracle,
            search_depth=search_depth,
            number_simulations=number_simulations,
            C_pw=C_pw,
            alpha_pw=alpha_pw,
            C_exp=C_exp,
            alpha_exp=alpha_exp,
            beta_policy=beta_policy,
            beta_value=beta_value,
            vis_on=vis_on
        )
    elif solver_name == "NeuralNetwork":
        solver = PolicySolver(policy_oracle=policy_oracle)

    return solver
