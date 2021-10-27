"""
Basic MCTS implementation
Graeme Best
Oregon State University
Jan 2020
"""


class Action:
    def __init__(self, name, label):
        self.name = name
        self.label = label

    def __str__(self):
        return str(self.label)


def print_action_sequence(action_sequence):
    sep = ''
    for action in action_sequence:
        print(f'{sep}action: {action}', end='')
        sep = ', '
    print()
