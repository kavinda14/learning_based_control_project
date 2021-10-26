'''
Basic MCTS implementation
Graeme Best
Oregon State University
Jan 2020
'''

from tree_node import TreeNode
import reward
from cost import cost
from rollout import rollout
from action import Action, printActionSequence
import copy
import random
import math


class State():
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
        if valid and new_loc not in sequence:
            neighbors.append(State(action, new_loc))

    return neighbors


# def mcts(action_set, budget, max_iterations, exploration_exploitation_parameter, robot, input_map):
def mcts(budget, max_iterations, exploration_exploitation_parameter, robot, sensor_model):

    ################################
    # Setup
    # what is a sequence?
    # -> a list to hold the actions, which can be used to calculate the budget left
    # in our case, this sequence is passed to every other node to keep track of history
    # the history will allow us to check if a particular state was previously visited
    # world_map = copy.deepcopy(input_map)
    start_sequence = [State('root', robot.get_loc())]
    # what is action_set?
    # -> there is an action object created in main.py
    unpicked_child_actions = generate_valid_neighbors(start_sequence[0], start_sequence, robot)
    # unpicked_child_actions = copy.deepcopy(action_set)
    # root is created when mcts is run
    root = TreeNode(parent=None, sequence=start_sequence, budget=budget, unpicked_child_actions=unpicked_child_actions, coords=robot.get_loc())
    list_of_all_nodes = []
    list_of_all_nodes.append(root) # for debugging only

    ################################
    # Main loop
    # this determines the depth of the tree
    for iter in range(max_iterations):

        # print('Iteration: ', iter)

        ################################
        # Selection and Expansion
        # move recursively down the tree from root
        # then add a new leaf node
        current = root
        while True: 

            # Are there any children to be added here?
            if current.unpicked_child_actions: # if not empty
                
                # does this mean that selection happens randomly?

                # Pick one of the children that haven't been added
                # Do this at random
                num_unpicked_child_actions = len(current.unpicked_child_actions)
                if num_unpicked_child_actions == 1:
                    child_index = 0
                else:
                    child_index = random.randint(0,num_unpicked_child_actions-1)
                child_action = current.unpicked_child_actions[child_index]
                child_loc = child_action.get_location()
                # print('child_loc: ', child_loc)

                # Remove the child form the unpicked list
                del current.unpicked_child_actions[child_index]

                # Setup the new action sequence
                new_sequence = copy.deepcopy(current.sequence)
                new_sequence.append(child_action)
                new_budget_left = budget - cost(new_sequence)

                # Setup the new child's unpicked children
                # Remove any over budget children from this set
                new_unpicked_child_actions = generate_valid_neighbors(child_action, new_sequence, robot)
                def is_overbudget(a):
                    seq_copy = copy.deepcopy(current.sequence)
                    seq_copy.append(a)
                    return cost(seq_copy) > budget
                
                # for the next node created, this adds actions to it only if there is budget left to do those actions
                new_unpicked_child_actions = [a for a in new_unpicked_child_actions if not is_overbudget(a)]

                # Create the new node and add it to the tree
                # printActionSequence(new_sequence)
                # EXPANSION
                new_child_node = TreeNode(parent=current, sequence=new_sequence, budget=new_budget_left, unpicked_child_actions=new_unpicked_child_actions, coords=child_loc)
                current.children.append(new_child_node)
                current = new_child_node
                list_of_all_nodes.append(new_child_node) # for debugging only

                break # don't go deeper in the tree...

            else:
                
                # All possible children already exist
                # Therefore recurse down the tree
                # using the UCT selection policy

                if not current.children:

                    # Reached planning horizon -- just do this again
                    break
                else:
                    # this is the point where the selection recurses down the tree till leaf node is reached

                    # Define the UCB
                    def ucb(average, n_parent, n_child):
                        return average + exploration_exploitation_parameter * math.sqrt( (2*math.log(n_parent)) / float(n_child) )

                    # Pick the child that maximises the UCB
                    n_parent = current.num_updates
                    best_child = -1
                    best_ucb_score = 0
                    for child_idx in range(len(current.children)):
                        child = current.children[child_idx]
                        ucb_score = ucb(child.average_evaluation_score, n_parent, child.num_updates)
                        if best_child == -1 or (ucb_score > best_ucb_score):
                            best_child = child
                            best_ucb_score = ucb_score

                    # Recurse down the tree
                    current = best_child

        ################################
        # Rollout
        # rollout_sequence = rollout(subsequence=current.sequence, action_set=action_set, budget=budget)
        rollout_sequence = rollout(subsequence=current.sequence, budget=budget, robot=robot)
        # rollout_reward = reward(action_sequence=rollout_sequence)
        rollout_reward = reward.greedy_reward(rollout_sequence, sensor_model)

        ################################
        # Back-propagation
        # update stats of all nodes from current back to root node
        parent = current
        while parent: # is not None

            # Update the average
            parent.updateAverage(rollout_reward)

            # Recurse up the tree
            parent = parent.parent

    ################################
    # Extract solution
    # calculate best solution so far
    # by recursively choosing child with highest average reward
    current = root
    # print('root :', current.get_children())
    # for node in current.get_children():
    #     print('child loc: ', node.get_coords())

    # while current.children: # is not empty

    #     # Find the child with best score
    #     best_score = 0
    #     best_child = -1
    #     for child_idx in range(len(current.children)):
    #         child = current.children[child_idx]
    #         score = child.average_evaluation_score
    #         if best_child == -1 or (score > best_score):
    #             best_child = child
    #             best_score = score

    #     current = best_child

    best_score = 0
    best_child = -1
    for child in current.children: # is not empty

        # Find the child with best score
        score = child.average_evaluation_score
        # print('score: ', [child.get_coords(), score])
        if best_child == -1 or (score > best_score):
            best_child = child
            best_score = score

    solution = current.sequence
    winner_node = best_child
    winner_loc = winner_node.get_coords()


    return [solution, root, list_of_all_nodes, winner_node, winner_loc]
