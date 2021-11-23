def get_reward(state, action: list):
    """
    Gets the reward for a secific state-action pair.
    State: [pos x, pos y, distance to goal, priority, 8 * normalized distance to closest agent, 8 * priorities]

    :param state: [vector len 21] State encoding of the agent
    :param action:  [vector len 8] Action taken by the agent
    :return:        scalar value based on reward calc
    """
    pos_x, pos_y = state[0], state[1]
    goal_dist = state[2]
    priority = state[3]

    reward = 1 / goal_dist
    # closest agents detected in all 8 octants
    for i in range(8):
        d_xi = state[3 + i]
        priority_i = state[11 + i]
        reward_i = (priority - priority_i) * (1 - d_xi)
        reward += reward_i

    return reward


def get_reward_old(priorities, goal, range, curr_state, all_agent_states, robot):
    """
    Calculate reward for agent at current state.
    priorities (list): goal priorities of all agents
    goal (tuple/list): x,y position of goal for current agent
    range (float?): radius of sensor range of current agent
    curr_state (tuple/list): x,y position of current agent
    all_agent_states (list of tuples/lists): x,y positions of all other agents
    robot (int): index of current agent
    :return: reward
    """

    x = robot
    p_x = priorities[x]
    rx = range
    n = len(priorities) - 1
    pos_x = curr_state
    d_xg = 0  # euclidean distance between pos_x and goal

    reward = 1 / d_xg
    for i in range(n):
        p_i = priorities[i]
        pos_i = all_agent_states[i]
        d_xi = 0  # euclidean distance between pos_x and pos_i

        indicator = 0
        if d_xi < range:
            indicator = 1

        reward_i = (p_x - p_i) * (1 - d_xi / rx) * indicator
        reward += reward_i

    return reward
