"""
@title

@description

"""
import math


def cost(action_sequence):
    return len(action_sequence)


def euclidean_distance(p1, p2):
    x1 = p1[0]
    y1 = p1[1]
    x2 = p2[0]
    y2 = p2[1]

    return math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
