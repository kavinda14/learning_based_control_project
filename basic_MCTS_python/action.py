'''
Basic MCTS implementation
Graeme Best
Oregon State University
Jan 2020
'''

class Action():
    def __init__(self, id, label):
        self.id = id
        self.label = label

    def toString(self):
        return str(self.label)

def printActionSequence(action_sequence):
    for action in action_sequence:
        print("action: ", action.toString() + ", "),
    print("")