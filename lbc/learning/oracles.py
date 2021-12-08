from lbc.learning.deterministic_policy_network import DeterministicPolicyNetwork
from lbc.learning.deterministic_value_network import DeterministicValueNetwork
from lbc.learning.gaussian_policy_network import GaussianPolicyNetwork
from lbc.learning.gaussian_value_network import GaussianValueNetwork
from lbc.problems.problem import Problem


def get_oracles(problem: Problem, policy_oracle_name=None, policy_oracle_paths=None, value_oracle_name=None,
                value_oracle_path=None, force=False):
    policy_oracle = [None for _ in range(problem.num_robots)]
    value_oracle = None

    if value_oracle_name == "deterministic" and (value_oracle_path is not None or force):
        value_oracle = DeterministicValueNetwork(problem, path=value_oracle_path)
    elif value_oracle_name == "gaussian" and (value_oracle_path is not None or force):
        value_oracle = GaussianValueNetwork(problem, path=value_oracle_path)

    if policy_oracle_name == "deterministic" and (any([a is not None for a in policy_oracle_paths]) or force):
        policy_oracle = [DeterministicPolicyNetwork(problem, path=a) for a in policy_oracle_paths]
    elif policy_oracle_name == "gaussian" and (any([a is not None for a in policy_oracle_paths]) or force):
        policy_oracle = [GaussianPolicyNetwork(problem, robot, path=a) for robot, a in enumerate(policy_oracle_paths)]

    return policy_oracle, value_oracle
