import math

import matplotlib.pyplot as plt


# def plotTree(list_of_all_nodes, winner, action_set, use_UCT, budget, fig_num, exploration_exploitation_parameter):
def plot_tree(list_of_all_nodes, winner, use_UCT, budget, fig_num, exploration_exploitation_parameter):
    def ucb(average, n_parent, n_child):
        return average + exploration_exploitation_parameter * math.sqrt((2 * math.log(n_parent)) / float(n_child))

    # Setup figure
    fig = plt.figure(fig_num)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()

    num_actions = 4
    # num_actions = len(action_set)

    # Compute colour bounds
    r_min = 1.0
    r_max = 0.0
    for n in list_of_all_nodes:
        if n.parent:
            if use_UCT:
                r = ucb(n.average_evaluation_score, n.parent.num_updates, n.num_updates)
            else:
                r = n.average_evaluation_score

            if r < r_min:
                r_min = r
            if r > r_max:
                r_max = r
    # Saturate a bit
    if use_UCT:
        f = 0.0
    else:
        f = 0.25
    r_min_new = r_min + f * (r_max - r_min)
    r_max_new = r_min + (1 - f) * (r_max - r_min)
    r_min = r_min_new
    r_max = r_max_new

    # Plot all nodes in the tree
    for i in range(len(list_of_all_nodes)):
        n = list_of_all_nodes[i]

        # Get position of this node
        my_depth = len(n.sequence)
        my_position = get_position(n.sequence, num_actions)

        # Plot edge to parent
        if n.parent:

            if use_UCT:
                r = ucb(n.average_evaluation_score, n.parent.num_updates, n.num_updates)
            else:
                r = n.average_evaluation_score

            # Normalise and saturate
            r = (r - r_min) / (r_max - r_min)
            if r < 0:
                r = 0.0
            if r > 1:
                r = 1.0

            # Generate colour based on reward
            col = (0, r, 0)

            # Get position of parent
            parent_depth = len(n.parent.sequence)
            parent_position = get_position(n.parent.sequence, num_actions)

            # Plot it
            x = (my_position, parent_position)
            y = (-my_depth, -parent_depth)
            ax.plot(x, y, color=col, zorder=1, linewidth=2)
            # printActionSequence(n.sequence)
            # print("(x,y): (" + str(my_position) + ", " + str(-my_depth) + ")")

            # Plot the winner as a circle
            if n == winner:
                x = my_position
                y = -my_depth
                winner_handle = ax.plot(x, y, 'or', zorder=2, linewidth=5, markersize=12)

    plt.axis('off')
    plt.show(block=False)


def get_position(seq, n):
    # Compute horizontal position based on sequence
    pos = -scramble(0, n)
    eps = 0
    maxn = 6

    for i in range(len(seq)):
        pos = max(((-eps / maxn) * i + eps), 1) * (pos + scramble(seq[i].name, n) * math.pow(n, -i))

    return -pos


def scramble(i, n):
    return -(i - math.floor(float(n) / 2.0)) / float(n)
