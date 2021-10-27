"""
Basic MCTS implementation
Graeme Best
Oregon State University
Jan 2020
"""

import random
import copy

from lbc.utils import cost


class State:
    def __init__(self, action, location):
        self.action = action
        self.location = location

    def get_action(self):
        return self.action

    def get_location(self):
        return self.location


def generate_valid_neighbors(current_state, state_sequence, robot):
    neighbors = list()
    current_loc = current_state.get_location()

    sequence = [state.get_location() for state in state_sequence]
    actions = ['left', 'right', 'forward', 'backward']
    for action in actions:
        valid, new_loc = robot.check_valid_move_mcts(action, current_loc, True)
        if valid:
            # if valid and new_loc not in sequence:
            # this makes the rollout not backtrack (might be too strict)
            # sequence.append(new_loc)
            neighbors.append(State(action, new_loc))

    return neighbors


def rollout(subsequence, budget, robot):
    # THESE ARE STATES
    current_state = subsequence[-1]
    current_loc = subsequence[-1].get_location()
    # print('rollout current_loc', current_loc)
    sequence = copy.deepcopy(subsequence)
    while cost(sequence) < budget:
        neighbors = generate_valid_neighbors(current_state, subsequence, robot)
        # for neighbor in neighbors:
        #     print('rollout neigh coords: ', neighbor.get_location())
        # print(len(neighbors))
        r = random.randint(0, len(neighbors) - 1)
        next_state = neighbors[r]
        sequence.append(next_state)
        current_loc = next_state.get_location()

    return sequence
