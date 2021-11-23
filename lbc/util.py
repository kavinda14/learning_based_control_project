import glob
import importlib
import os
import pickle
from queue import Empty

import numpy as np
from tqdm import tqdm

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

SOLVER_MAP = {
    'empty': Empty,
    'mcts': MCTS,
    'puct_v1': PUCT_V1,
    'puct_v2': PUCT_V2,
    'NeuralNetwork': PolicySolver
}


def get_problem_names():
    return PROBLEM_MAP.keys()


def get_problem(problem_name):
    problem = PROBLEM_MAP[problem_name.lower()]
    return problem()


def get_solver(solver_name, vis_on=False, **kwargs):
    """
    policy_oracle=None, value_oracle=None, search_depth=10, number_simulations=1000,
    C_pw=2.0, alpha_pw=0.5, C_exp=1.0, alpha_exp=0.25, beta_policy=0.0, beta_value=1.0,

    :param solver_name:
    :param vis_on:
    :param kwargs:
    :return:
    """
    solver = SOLVER_MAP[solver_name.lower()]
    solver = solver(**kwargs)
    return solver
    # if solver_name == "Empty":
    #     solver = Empty()
    # elif solver_name == "MCTS":
    #     solver = MCTS()
    # elif solver_name == "PUCT_V1":
    #     solver = PUCT_V1(
    #         policy_oracle=policy_oracle,
    #         value_oracle=value_oracle,
    #         search_depth=search_depth,
    #         number_simulations=number_simulations,
    #         C_pw=C_pw,
    #         alpha_pw=alpha_pw,
    #         C_exp=C_exp,
    #         alpha_exp=alpha_exp,
    #         beta_policy=beta_policy,
    #         beta_value=beta_value,
    #         vis_on=vis_on
    #     )
    # elif solver_name == "PUCT_V2":
    #     solver = PUCT_V2(
    #         policy_oracle=policy_oracle,
    #         value_oracle=value_oracle,
    #         search_depth=search_depth,
    #         number_simulations=number_simulations,
    #         C_pw=C_pw,
    #         alpha_pw=alpha_pw,
    #         C_exp=C_exp,
    #         alpha_exp=alpha_exp,
    #         beta_policy=beta_policy,
    #         beta_value=beta_value,
    #         vis_on=vis_on
    #     )
    # elif solver_name == "NeuralNetwork":
    #     solver = PolicySolver(policy_oracle=policy_oracle)

    return solver


def dbgp(name, value):
    if type(value) is dict:
        print('{}'.format(name))
        for key_i, value_i in value.items():
            print('{}:{}'.format(str(key_i), value_i))
    else:
        print('{}:{}'.format(name, value))


def load_module(fn):
    module_dir, module_name = fn.split("/")
    module_name, _ = module_name.split(".")
    module = importlib.import_module("{}.{}".format(module_dir, module_name))
    return module


def write_sim_result(sim_result_dict, fn):
    with open(fn + '.pickle', 'xb') as h:
        pickle.dump(sim_result_dict, h)


def load_sim_result(fn):
    with open(fn, 'rb') as h:
        sim_result = pickle.load(h)
    return sim_result


def write_dataset(dataset, fn):
    # with open(fn, 'xb') as h:
    # 	pickle.dump(dataset, h)
    np.save(fn, dataset)


def get_dataset_fn(oracle_name, l, robot=0):
    # return "../current/data/{}_l{}_i{}.pickle".format(oracle,l,robot)
    return "../current/data/{}_l{}_i{}.npy".format(oracle_name, l, robot)


def get_oracle_fn(l, num_robots):
    value_oracle_path = "../current/models/model_value_l{}.pt".format(l)
    policy_oracle_paths = []
    for i in range(num_robots):
        policy_oracle_paths.append("../current/models/model_policy_l{}_i{}.pt".format(l, i))
    return value_oracle_path, policy_oracle_paths


def format_dir(clean_dirnames):
    dirnames = ["plots", "data", "models"]
    for dirname in dirnames:
        path = os.path.join(os.getcwd(), "../current/{}".format(dirname))
        os.makedirs(path, exist_ok=True)
    for dirname in clean_dirnames:
        path = os.path.join(os.getcwd(), "../current/{}".format(dirname))
        for file in glob.glob(path + "/*"):
            os.remove(file)


def get_temp_fn(dirname, i):
    return "{}/temp_{}.npy".format(dirname, i)


def init_tqdm(rank, total):
    pbar = None
    if rank == 0:
        pbar = tqdm(total=total)
    return pbar


def update_tqdm(rank, total_per_worker, queue, pbar):
    if rank == 0:
        count = total_per_worker
        try:
            while True:
                count += queue.get_nowait()
        except Empty:
            pass
        pbar.update(count)
    else:
        queue.put_nowait(total_per_worker)
