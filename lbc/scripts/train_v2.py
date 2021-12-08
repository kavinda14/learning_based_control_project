import tempfile
import time

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from lbc import plotter
from lbc.learning.oracles import get_oracles
from lbc.problems.problem import Problem
from lbc.util import get_oracle_fn, format_dir, get_solver, get_problem, datapoints_to_dataset


# expert value demonstration functions
def worker_edv(solver, save_path, seed, problem: Problem, num_states):
    np.random.seed(seed)
    pbar = tqdm(total=num_states)
    datapoints = []
    while len(datapoints) < num_states:
        root_state = problem.sample_state()
        root_node = solver.search(problem, root_state)
        if root_node.success:
            datapoint = np.append(root_state.squeeze(), root_node.value)
            datapoints.append(datapoint)
            pbar.update(1)
    datapoints = np.array(datapoints)
    np.save(save_path, datapoints)
    return datapoints


def make_expert_demonstration_v(problem: Problem, solver_name, num_states, value_oracle, policy_oracle, train_size,
                                learning_idx,
                                number_simulations, beta_value):
    start_time = time.time()
    print('making value dataset...')

    _, save_path = tempfile.mkstemp()
    seed = np.random.randint(10000)
    # policy_oracle=None, value_oracle=None, search_depth=10, number_simulations=1000, C_pw=2.0, alpha_pw=0.5,
    # C_exp=1.0, alpha_exp=0.25, beta_policy=0., beta_value=0., vis_on=False
    solver = get_solver(solver_name, policy_oracle=policy_oracle, value_oracle=value_oracle,
                        number_simulations=number_simulations, beta_value=beta_value)
    datapoints = worker_edv(solver, save_path, seed, problem, num_states)

    split = int(len(datapoints) * train_size)
    train_dataset = datapoints_to_dataset(datapoints[0:split], "train_value",
                                          encoding_dim=problem.value_encoding_dim, learning_idx=learning_idx)
    test_dataset = datapoints_to_dataset(datapoints[split:], "test_value",
                                         encoding_dim=problem.value_encoding_dim, learning_idx=learning_idx)
    plotter.plot_value_dataset(problem,
                               [[train_dataset.X_np, train_dataset.target_np],
                                [test_dataset.X_np, test_dataset.target_np]],
                               ["Train", "Test"])
    plotter.save_figs(f"../current/models/dataset_value_l{learning_idx}.pdf")
    print(f'expert demonstration v completed in {time.time() - start_time}s.')
    return train_dataset, test_dataset


def train_model(problem: Problem, train_dataset, test_dataset, learning_idx, oracle_type, value_oracle_name,
                learning_rate, batch_size, num_epochs, device, robot=0):
    start_time = time.time()
    print('training model...')

    model_fn, _ = get_oracle_fn(learning_idx, problem.num_robots)
    _, model = get_oracles(problem, value_oracle_name=value_oracle_name, force=True)
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=50, min_lr=1e-4, verbose=True)

    train_dataset.to(device)
    test_dataset.to(device)
    # noinspection PyUnresolvedReferences
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size)
    # noinspection PyUnresolvedReferences
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)

    losses = []
    best_test_loss = np.Inf
    for _ in tqdm(range(num_epochs)):
        train_epoch_loss = train(model, optimizer, train_loader)
        test_epoch_loss = test(model, test_loader)
        scheduler.step(test_epoch_loss)
        losses.append((train_epoch_loss, test_epoch_loss))
        if test_epoch_loss < best_test_loss:
            best_test_loss = test_epoch_loss
            torch.save(model.to(device).state_dict(), model_fn)
            model.to(device)
    plotter.plot_loss(losses)
    plotter.save_figs(f"../current/models/losses_{oracle_type}_l{learning_idx}_i{robot}.pdf")
    print(f'training model completed in {time.time() - start_time}s.')
    return


def train(model, optimizer, loader):
    epoch_loss = 0
    for step, (x, target) in enumerate(loader):
        loss = model.loss_fnc(x, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += float(loss)
    return epoch_loss / len(loader)


def test(model, loader):
    epoch_loss = 0
    for step, (x, target) in enumerate(loader):
        loss = model.loss_fnc(x, target)
        epoch_loss += float(loss)
    return epoch_loss / len(loader)


def eval_value(problem: Problem, learning_idx, num_v_eval, value_oracle_name):
    value_oracle_path, policy_oracle_paths = get_oracle_fn(learning_idx, problem.num_robots)
    _, value_oracle = get_oracles(problem,
                                  value_oracle_name=value_oracle_name,
                                  value_oracle_path=value_oracle_path
                                  )

    states = []
    values = []
    for _ in range(num_v_eval):
        state = problem.sample_state()
        value = value_oracle.eval(problem, state)
        states.append(state)
        values.append(value)

    states = np.array(states).squeeze(axis=2)
    values = np.array(values).squeeze(axis=2)
    plotter.plot_value_dataset(problem, [[states, values]], ["Eval"])
    plotter.save_figs(f"../current/models/value_eval_l{learning_idx}.pdf")
    return


def main():
    # solver settings
    number_simulations = 50
    search_depth = 50
    alpha_pw = 0.25
    alpha_exp = 0.5
    beta_policy = 0.0
    beta_value = 0.7
    vis_on = True
    device = 'cuda'

    solver_name = "PUCT_V1"
    problem_name = "lbc_simple"
    value_oracle_name = "deterministic"

    # learning
    num_learning_iters = 20
    num_d_v = 20
    num_v_eval = 20

    learning_rate = 0.001
    num_epochs = 100
    batch_size = 128
    train_size = 0.8

    problem = get_problem(problem_name)
    format_dir(clean_dirnames=["data", "models"])

    if batch_size > num_d_v * (1 - train_size):
        batch_size = int(num_d_v * train_size / 5)
        print(f'changing batch size to {batch_size}')

    # training
    for learning_idx in range(num_learning_iters):
        start_time = time.time()
        print(f'learning iteration: {learning_idx}/{num_learning_iters}...')

        policy_oracle = [None for _ in range(problem.num_robots)]

        if learning_idx == 0:
            value_oracle_path = None
            value_oracle = None
        else:
            value_oracle_path, _ = get_oracle_fn(learning_idx - 1, problem.num_robots)
            _, value_oracle = get_oracles(problem,
                                          value_oracle_name=value_oracle_name,
                                          value_oracle_path=value_oracle_path
                                          )

        print(f'\tvalue training: {learning_idx}/{num_learning_iters}')
        train_dataset_v, test_dataset_v = make_expert_demonstration_v(
            problem, solver_name=solver_name, num_states=num_d_v,
            value_oracle=value_oracle, policy_oracle=policy_oracle, train_size=train_size, learning_idx=learning_idx,
            beta_value=beta_value, number_simulations=number_simulations
        )
        train_model(
            problem, train_dataset_v, test_dataset_v,
            learning_idx=learning_idx, oracle_type="value",
            device=device, batch_size=batch_size, learning_rate=learning_rate,
            num_epochs=num_epochs, value_oracle_name=value_oracle_name
        )
        eval_value(problem, learning_idx, num_v_eval=num_v_eval, value_oracle_name=value_oracle_name)
        print(f'complete learning iteration: {learning_idx}/{num_learning_iters} in {time.time() - start_time}s')
    return


if __name__ == '__main__':
    main()
