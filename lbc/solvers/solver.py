from lbc.solvers.empty import Empty
from lbc.solvers.mcts import MCTS
from lbc.solvers.policy_solver import PolicySolver
from lbc.solvers.puct_v1 import PUCT_V1
from lbc.solvers.puct_v2 import PUCT_V2


class Solver:

    def __init__(self):
        pass

    def policy(self, problem, state):
        # output:
        # 	- action: [nd x 1] array
        exit("policy not overwritten")


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
