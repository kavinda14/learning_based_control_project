"""
Basic MCTS implementation
Graeme Best
Oregon State University
Jan 2020
"""


class TreeNode:
    def __init__(self, parent, sequence, budget, unpicked_child_actions, coords):
        # tree properties
        self.parent = parent
        self.children = []
        self.unpicked_child_actions = unpicked_child_actions
        self.coords = coords

        # sequence properties
        self.sequence = sequence
        self.budget = budget

        # reward estimate properties
        self.average_evaluation_score = 0
        self.num_updates = 0

    def update_average(self, evaluation_score):
        # Incremental update to the average
        self.average_evaluation_score = float(
            self.average_evaluation_score * self.num_updates + evaluation_score) / float(self.num_updates + 1)
        self.num_updates = self.num_updates + 1

    def get_coords(self):
        return self.coords

    def get_children(self):
        return self.children


def count_nodes(current):
    count = 1
    for child in current.children:
        count += count_nodes(child)
    return count
