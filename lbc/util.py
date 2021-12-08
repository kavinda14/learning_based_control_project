import glob
import importlib
import os
import pickle
from queue import Empty

import numpy as np
import torch
from tqdm import tqdm

from lbc.problems.lbc_examples.lbc_simple import LbcSimple
from lbc.problems.nte_examples.example1 import Example1
from lbc.problems.nte_examples.example10 import Example10
from lbc.problems.nte_examples.example11 import Example11
from lbc.problems.nte_examples.example12 import Example12
from lbc.problems.nte_examples.example2 import Example2
from lbc.problems.nte_examples.example3 import Example3
from lbc.problems.nte_examples.example4 import Example4
from lbc.problems.nte_examples.example5 import Example5
from lbc.problems.nte_examples.example6 import Example6
from lbc.problems.nte_examples.example7 import Example7
from lbc.problems.nte_examples.example8 import Example8
from lbc.problems.nte_examples.example9 import Example9
from lbc.solvers.policy_solver import PolicySolver
from lbc.solvers.puct_v1 import PUCT_V1

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
    'lbc_simple': LbcSimple,  # modified homicidal chauffer
}

SOLVER_MAP = {
    'puct_v1': PUCT_V1,
    'neuralnetwork': PolicySolver
}


def get_problem_names():
    return PROBLEM_MAP.keys()


def get_problem(problem_name, **kwargs):
    problem = PROBLEM_MAP[problem_name.lower()]
    return problem(**kwargs)


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
    return "../current_bak_2/data/{}_l{}_i{}.npy".format(oracle_name, l, robot)


def get_oracle_fn(l, num_robots):
    value_oracle_path = "../current_bak_2/models/model_value_l{}.pt".format(l)
    policy_oracle_paths = []
    for i in range(num_robots):
        policy_oracle_paths.append("../current_bak_2/models/model_policy_l{}_i{}.pt".format(l, i))
    return value_oracle_path, policy_oracle_paths


def format_dir(clean_dirnames):
    dirnames = ["plots", "data", "models"]
    for dirname in dirnames:
        path = os.path.join(os.getcwd(), "../current_bak_2/{}".format(dirname))
        os.makedirs(path, exist_ok=True)
    for dirname in clean_dirnames:
        path = os.path.join(os.getcwd(), "../current_bak_2/{}".format(dirname))
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


def datapoints_to_dataset(datapoints, oracle_name, encoding_dim, learning_idx, robot=0):
    dataset_fn = get_dataset_fn(oracle_name, learning_idx, robot=robot)
    datapoints = np.array(datapoints)
    write_dataset(datapoints, dataset_fn)
    dataset = Dataset(dataset_fn, encoding_dim, device='cuda')
    return dataset


# noinspection PyUnresolvedReferences
class Dataset(torch.utils.data.Dataset):

    def __init__(self, src_file, encoding_dim, device='cpu'):
        datapoints = np.load(src_file)
        self.X_np, self.target_np = datapoints[:, 0:encoding_dim], datapoints[:, encoding_dim:]
        self.X_torch = torch.tensor(self.X_np, dtype=torch.float32, device=device)
        self.target_torch = torch.tensor(self.target_np, dtype=torch.float32, device=device)
        return

    def __len__(self):
        return self.X_torch.shape[0]

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.X_torch[idx, :], self.target_torch[idx, :]

    def to(self, device):
        self.X_torch = self.X_torch.to(device)
        self.target_torch = self.target_torch.to(device)
        return
