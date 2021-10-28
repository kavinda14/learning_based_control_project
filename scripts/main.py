"""
Basic MCTS implementation
Graeme Best
Oregon State University
Jan 2020
"""

import sys
import time

from lbc.mcts.action import Action, print_action_sequence
from lbc.mcts.mcts import mcts
from scripts.plot_tree import plotTree


def run():
    # Setup the problem
    num_actions = 3
    action_set = []
    for i in range(num_actions):
        action_set.append(Action(i, i))
    budget = 7

    # Solve it with ACTS
    # =1.0 is recommended. <1.0 more exploitation. >1.0 more exploration
    exploration_exploitation_parameter = 0.8
    max_iterations = 2000
    [solution, root, list_of_all_nodes, winner] = mcts(action_set, budget, max_iterations,
                                                       exploration_exploitation_parameter)

    # Display the tree
    print_action_sequence(solution)
    plotTree(list_of_all_nodes, winner, action_set, False, budget, 1, exploration_exploitation_parameter)
    plotTree(list_of_all_nodes, winner, action_set, True, budget, 2, exploration_exploitation_parameter)

    # Wait for Ctrl+C
    while True:
        try:
            time.sleep(.1)
        except KeyboardInterrupt:
            sys.exit()


if __name__ == "__main__":
    run()
