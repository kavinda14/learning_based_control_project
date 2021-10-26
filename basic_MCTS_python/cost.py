'''
Basic MCTS implementation
Graeme Best
Oregon State University
Jan 2020
'''

def cost(action_sequence):
    # A simple cost evaluation function
    return len(action_sequence)