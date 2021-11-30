import os
import time
from queue import Queue

import numpy as np
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm

from lbc import plotter
from lbc.learning.oracles import get_oracles
from lbc.problems.problem import Problem
from lbc.scripts.run import run_instance
from lbc.util import get_temp_fn, get_oracle_fn, format_dir, get_problem, get_solver, datapoints_to_dataset


def worker_edp(seed, fn, problem: Problem, robot, num_per_pool, solver, mode, num_subsamples):
    np.random.seed(seed)
    datapoints = []

    robot_action_idx = problem.action_idxs[robot]
    count = 0
    pbar = tqdm(total=num_per_pool)
    while count < num_per_pool:
        state = problem.initialize()
        root_node = solver.search(problem, state, turn=robot)
        if root_node.success:
            encoding = problem.policy_encoding(state, robot).squeeze()
            if mode == 0:
                # weighted average of children
                actions, num_visits = solver.get_child_distribution(root_node)
                robot_actions = np.array(actions)[:, robot_action_idx]
                target = np.average(robot_actions, weights=num_visits, axis=0)
                datapoint = np.append(encoding, target)
                datapoints.append(datapoint)
            elif mode == 1:
                # best child
                actions, num_visits = solver.get_child_distribution(root_node)
                target = np.array(actions[np.argmax(num_visits)])[robot_action_idx]
                datapoint = np.append(encoding, target)
                datapoints.append(datapoint)
            elif mode == 2:
                # subsampling of children method
                actions, num_visits = solver.get_child_distribution(root_node)
                choice_idxs = np.random.choice(len(actions), num_subsamples, p=num_visits / np.sum(num_visits))
                for choice_idx in choice_idxs:
                    target = np.array(actions[choice_idx])[robot_action_idx]
                    datapoint = np.append(encoding, target)
                    datapoints.append(datapoint)
            count += 1
            pbar.update(1)
    np.save(fn, np.array(datapoints))
    return datapoints


def make_expert_demonstration_pi(problem: Problem, robot, solver, train_size, learning_idx,
                                 num_D_pi, num_subsamples, mode, dirname):
    start_time = time.time()
    print('making expert demonstration pi...')

    paths = []
    path = get_temp_fn(dirname, 0)
    seed = np.random.randint(10000)
    paths.append(path)
    worker_edp(seed, path, problem, robot, num_D_pi, solver=solver, mode=mode, num_subsamples=num_subsamples)

    datapoints = []
    for path in paths:
        datapoints.extend(list(np.load(path)))
        os.remove(path)

    split = int(len(datapoints) * train_size)
    train_dataset = datapoints_to_dataset(datapoints[0:split], "train_policy",
                                          encoding_dim=problem.policy_encoding_dim,
                                          learning_idx=learning_idx, robot=robot)
    test_dataset = datapoints_to_dataset(datapoints[split:], "test_policy",
                                         encoding_dim=problem.policy_encoding_dim,
                                         learning_idx=learning_idx, robot=robot)
    plotter.plot_policy_dataset(problem,
                                [[train_dataset.X_np, train_dataset.target_np],
                                 [test_dataset.X_np, test_dataset.target_np]],
                                ["Train", "Test"], robot)
    plotter.save_figs("{}/dataset_policy_l{}_i{}.pdf".format(dirname, learning_idx, robot))
    print('expert demonstration pi completed in {}s.'.format(time.time() - start_time))
    return train_dataset, test_dataset


def worker_edv(fn, seed, problem: Problem, num_states_per_pool, policy_oracle):
    solver = get_solver(
        "NeuralNetwork",
        policy_oracle=policy_oracle)

    instance = {
        "problem": problem,
        "solver": solver,
    }
    np.random.seed(seed)

    datapoints = []
    pbar = tqdm(total=num_states_per_pool)
    while len(datapoints) < num_states_per_pool:
        state = problem.initialize()
        instance["initial_state"] = state
        sim_result = run_instance(0, Queue(), 0, instance, verbose=False, tqdm_on=False)
        value = calculate_value(problem, sim_result)
        encoding = problem.value_encoding(state).squeeze()
        datapoint = np.append(encoding, value)
        datapoints.append(datapoint)
        pbar.update(1)
    np.save(fn, np.array(datapoints))
    return datapoints


def make_expert_demonstration_v(problem: Problem, learning_idx, train_size, policy_oracle_name, num_D_v, dirname):
    start_time = time.time()
    print('making value dataset...')

    _, policy_oracle_paths = get_oracle_fn(learning_idx, problem.num_robots)
    policy_oracle, _ = get_oracles(problem,
                                   policy_oracle_name=policy_oracle_name,
                                   policy_oracle_paths=policy_oracle_paths
                                   )

    paths = [get_temp_fn(dirname, 0)]
    seed = np.random.randint(10000)
    worker_edv(paths[0], seed, problem, num_D_v, policy_oracle)

    datapoints = []
    for path in paths:
        datapoints_i = np.load(path, allow_pickle=True)
        datapoints.extend(datapoints_i)
        os.remove(path)

    split = int(len(datapoints) * train_size)
    train_dataset = datapoints_to_dataset(datapoints[0:split], "train_value",
                                          encoding_dim=problem.value_encoding_dim, learning_idx=learning_idx)
    test_dataset = datapoints_to_dataset(datapoints[split:], "test_value",
                                         encoding_dim=problem.value_encoding_dim, learning_idx=learning_idx)
    plotter.plot_value_dataset(problem,
                               [[train_dataset.X_np, train_dataset.target_np],
                                [test_dataset.X_np, test_dataset.target_np]],
                               ["Train", "Test"])
    plotter.save_figs("{}/dataset_value_l{}.pdf".format(dirname, learning_idx))
    print('expert demonstration v completed in {}s.'.format(time.time() - start_time))
    return train_dataset, test_dataset


def calculate_value(problem: Problem, sim_result):
    value = np.zeros((problem.num_robots, 1))
    states = sim_result["states"]
    actions = sim_result["actions"]
    for step, (state, action) in enumerate(zip(states, actions)):
        reward = problem.normalized_reward(state, action)
        value += (problem.gamma ** step) * reward
    return value


def train_model(problem: Problem, train_dataset, test_dataset, learning_idx, policy_oracle_name, value_oracle_name,
                oracle_type, learning_rate, batch_size, num_epochs, dirname, robot=0, device='cuda'):
    start_time = time.time()
    print('training model...')

    value_oracle_path, policy_oracle_paths = get_oracle_fn(learning_idx, problem.num_robots)

    if oracle_type == "policy":
        model_fn = policy_oracle_paths[robot]
        model, _ = get_oracles(problem,
                               policy_oracle_name=policy_oracle_name,
                               policy_oracle_paths=[None for _ in range(problem.num_robots)],
                               force=True
                               )
        model = model[robot]
    elif oracle_type == "value":
        model_fn = value_oracle_path
        _, model = get_oracles(problem,
                               value_oracle_name=value_oracle_name,
                               force=True
                               )
    else:
        raise ValueError(f'Unrecognized value for parameter oracle_name: {oracle_type}')
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, 'min',
                                  factor=0.5, patience=50, min_lr=1e-4, verbose=True)

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
            torch.save(model.to('cpu').state_dict(), model_fn)
            model.to(device)
    plotter.plot_loss(losses)
    plotter.save_figs("{}/losses_{}_l{}_i{}.pdf".format(dirname, oracle_type, learning_idx, robot))
    print('training model completed in {}s.'.format(time.time() - start_time))
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


def eval_value(problem: Problem, learning_idx, value_oracle_name, dirname, num_v_eval):
    value_oracle_path, policy_oracle_paths = get_oracle_fn(learning_idx, problem.num_robots)

    _, value_oracle = get_oracles(problem,
                                  value_oracle_name=value_oracle_name,
                                  value_oracle_path=value_oracle_path
                                  )

    states = []
    values = []
    encodings = []
    for _ in range(num_v_eval):
        state = problem.initialize()
        encoding = problem.value_encoding(state)
        value = value_oracle.eval(problem, state)
        states.append(state)
        values.append(value)
        encodings.append(encoding.reshape((problem.value_encoding_dim, 1)))

    # states = np.array(states).squeeze(axis=2)
    # values = np.array(values).squeeze(axis=2)
    # encodings = np.array(encodings).squeeze(axis=2)
    # plotter.plot_value_dataset(problem, [[encodings, values]], ["Eval"])
    # plotter.save_figs("{}/value_eval_l{}.pdf".format(dirname, learning_idx))
    return


def eval_policy(problem: Problem, learning_idx, policy_oracle_name, num_pi_eval, dirname, robot):
    value_oracle_path, policy_oracle_paths = get_oracle_fn(learning_idx, problem.num_robots)
    for robot_i, path in enumerate(policy_oracle_paths):
        if robot_i != robot:
            policy_oracle_paths[robot_i] = None

    policy_oracles, _ = get_oracles(problem,
                                    policy_oracle_name=policy_oracle_name,
                                    policy_oracle_paths=policy_oracle_paths
                                    )
    policy_oracle = policy_oracles[robot]

    states = []
    encodings = []
    actions = []
    robot_action_dim = len(problem.action_idxs[robot])
    for _ in range(num_pi_eval):
        state = problem.initialize()
        encoding = problem.policy_encoding(state, robot)
        encoding = torch.tensor(encoding, dtype=torch.float32).squeeze().unsqueeze(0)  # [batch_size x state_dim]
        mu, logvar = policy_oracle(encoding, training=True)  # mu in [1 x robot_action_dim]
        mu = mu.detach().numpy().reshape((robot_action_dim, 1))
        sd = np.sqrt(np.exp(logvar.detach().numpy().reshape((robot_action_dim, 1))))
        action = np.concatenate((mu, sd), axis=0)
        states.append(state)
        actions.append(action)
        encodings.append(encoding.detach().numpy().reshape((problem.policy_encoding_dim, 1)))

    # states = np.array(states).squeeze(axis=2)
    # actions = np.array(actions).squeeze(axis=2)
    # encodings = np.array(encodings).squeeze(axis=2)
    # plotter.plot_policy_dataset(problem,[[states,actions]],["Eval"],robot)
    # plotter.plot_policy_dataset(problem, [[encodings, actions]], ["Eval"], robot)
    # plotter.save_figs("{}/policy_eval_l{}_i{}.pdf".format(dirname, learning_idx, robot))
    return


def self_play(problem: Problem, policy_oracle, value_oracle, learning_idx, num_self_play_plots, dirname):
    solver = get_solver(
        "NeuralNetwork",
        policy_oracle=policy_oracle)

    instance = {
        "problem": problem,
        "solver": solver,
        "policy_oracle": policy_oracle,
        "value_oracle": value_oracle,
    }

    sim_results = []
    for _ in range(num_self_play_plots):
        state = problem.initialize()
        instance["initial_state"] = state
        sim_result = run_instance(0, Queue(), 0, instance, verbose=False, tqdm_on=False)
        sim_results.append(sim_result)

    for sim_result in sim_results:
        plotter.plot_sim_result(sim_result)
        problem.render(states=sim_result["states"])

    if hasattr(problem, 'pretty_plot'):
        problem.pretty_plot(sim_results[0])

    plotter.save_figs("{}/self_play_l{}.pdf".format(dirname, learning_idx))
    return sim_results


def main():
    num_simulations = 20
    search_depth = 5
    c_pw = 2.0
    alpha_pw = 0.5
    c_exp = 1.0
    alpha_exp = 0.25
    beta_policy = 0.5
    beta_value = 0.5
    solver_name = "PUCT_V1"
    problem_name = "lbc_simple"
    policy_oracle_name = "gaussian"
    value_oracle_name = "deterministic"

    dirname = "../current/models"

    # learning
    learning_iters = 40
    # 0: weighted sum, 1: best child, 2: subsamples
    mode = 1
    num_d_pi = 20
    num_pi_eval = 20
    num_d_v = 20
    num_v_eval = 20
    num_subsamples = 5
    num_self_play_plots = 10
    learning_rate = 0.001
    num_epochs = 20
    batch_size = 1028
    train_size = 0.8

    problem: Problem = get_problem(problem_name)
    format_dir(clean_dirnames=["data", "models"])

    num_d_pi_samples = num_d_pi
    if mode == 2:
        num_d_pi_samples = num_d_pi * num_subsamples
    if batch_size > np.min((num_d_pi_samples, num_d_v)) * (1 - train_size):
        batch_size = int(np.floor((np.min((num_d_pi_samples, num_d_v)) * train_size / 10)))
        print(f'changing batch size to {batch_size}')

    # training
    for learning_idx in range(learning_iters):
        start_time = time.time()
        print(f'learning iteration: {learning_idx}/{learning_iters}')

        if learning_idx == 0:
            policy_oracle = [None for _ in range(problem.num_robots)]
            value_oracle = None
        else:
            value_oracle_path, policy_oracle_paths = get_oracle_fn(learning_idx - 1, problem.num_robots)
            policy_oracle, value_oracle = get_oracles(problem,
                                                      value_oracle_name=value_oracle_name,
                                                      value_oracle_path=value_oracle_path,
                                                      policy_oracle_name=policy_oracle_name,
                                                      policy_oracle_paths=policy_oracle_paths
                                                      )

            print(f'\t self play l/L: {learning_idx}/{learning_iters}')
            sim_results = self_play(problem, policy_oracle=policy_oracle, value_oracle=value_oracle,
                                    learning_idx=learning_idx - 1,
                                    dirname=dirname, num_self_play_plots=num_self_play_plots)

        solver = get_solver(
            solver_name, policy_oracle=policy_oracle, value_oracle=value_oracle,
            search_depth=search_depth, number_simulations=num_simulations,
            C_pw=c_pw, alpha_pw=alpha_pw, C_exp=c_exp, alpha_exp=alpha_exp,
            beta_policy=beta_policy, beta_value=beta_value
        )

        for robot in range(problem.num_robots):
            print(f'\tpolicy training iteration l/L, i/N: {learning_idx}/{learning_iters} {robot}/{problem.num_robots}')
            train_dataset_pi, test_dataset_pi = make_expert_demonstration_pi(
                problem, robot=robot, solver=solver, train_size=train_size, learning_idx=learning_idx,
                dirname=dirname, mode=mode, num_D_pi=num_d_pi, num_subsamples=num_subsamples)
            train_model(problem, train_dataset_pi, test_dataset_pi, learning_idx=learning_idx, oracle_type="policy",
                        robot=robot, batch_size=batch_size, num_epochs=num_epochs, learning_rate=learning_rate,
                        policy_oracle_name=policy_oracle_name, value_oracle_name=value_oracle_name, dirname=dirname)
            eval_policy(problem, learning_idx=learning_idx, robot=robot, dirname=dirname,
                        policy_oracle_name=policy_oracle_name, num_pi_eval=num_pi_eval)

        print(f'\t value training l/L: {learning_idx}/{learning_iters}')
        train_dataset_v, test_dataset_v = make_expert_demonstration_v(
            problem, learning_idx=learning_idx, train_size=train_size, policy_oracle_name=policy_oracle_name,
            num_D_v=num_d_v, dirname=dirname)
        train_model(problem, train_dataset_v, test_dataset_v, learning_idx=learning_idx,
                    policy_oracle_name=policy_oracle_name, value_oracle_name=value_oracle_name, oracle_type="value",
                    learning_rate=learning_rate, batch_size=batch_size, num_epochs=num_epochs, dirname=dirname)
        eval_value(problem, learning_idx=learning_idx, num_v_eval=num_v_eval,
                   value_oracle_name=value_oracle_name, dirname=dirname)
        print(f'complete learning iteration: {learning_idx}/{learning_iters} in {time.time() - start_time}s')
    return


if __name__ == '__main__':
    main()
