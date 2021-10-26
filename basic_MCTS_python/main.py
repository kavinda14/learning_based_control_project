'''
Basic MCTS implementation
Graeme Best
Oregon State University
Jan 2020
'''

from mcts import mcts
from action import Action, printActionSequence
from tree_node import countNodes
from plot_tree import plotTree
import time, sys

def run():
    # Setup the problem
    num_actions = 3
    action_set = []
    for i in range(num_actions):
        id = i
        action_set.append(Action(id,i))
    budget = 7
    

    # Solve it with MCTS
    exploration_exploitation_parameter = 0.8 # =1.0 is recommended. <1.0 more exploitation. >1.0 more exploration. 
    max_iterations = 2000
    [solution, root, list_of_all_nodes, winner] = mcts( action_set, budget, max_iterations, exploration_exploitation_parameter )

    # Display the tree
    printActionSequence(solution)
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
    
    