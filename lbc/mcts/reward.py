"""
Basic MCTS implementation
Graeme Best
Oregon State University
Jan 2020
"""

import random


def reward(sequence):
    return random.randint(0, 10) + len(sequence)


def greedy_reward(rollout_sequence, sensor_model):
    reward_val = 0

    for state in rollout_sequence:
        state_loc = state.get_location()
        # print(state_loc)
        # scanned_unobs = sensor_model.scan(state_loc, False)
        reward_val += len(sensor_model.scan(state_loc, False)[0])
        # reward += len(scanned_unobs[0]) + len(scanned_unobs[1])
        # print('reward: ', reward)
    return reward_val
