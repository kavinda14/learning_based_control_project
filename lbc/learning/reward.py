"""
Reward functions for lbc agent.
"""
import math


def distance(start_coord, end_coord, degree=2):
    if len(start_coord) != len(end_coord):
        raise ArithmeticError(f'Unable to compute distance between vectors of different dimensions: '
                              f'start: {len(start_coord)} -> end: {len(end_coord)}')
    dist = 0
    for each_start, each_end in zip(start_coord, end_coord):
        inner_dist = each_start - each_end
        inner_dist = math.pow(inner_dist, degree)
        dist += inner_dist
    dist = math.pow(dist, 1/degree)
    return dist


def get_reward(state, action: list, num_regions=8):
    """
    Gets the reward for a secific state-action pair.
    State: [pos x, pos y, distance to goal, priority, 8 * normalized distance to closest agent, 8 * priorities]

    :param state:       [vector len 21] State encoding of the agent
    :param action:      [vector len 8] Action taken by the agent
    :param num_regions: number of regions around the agent
    :return:            reward for this state
    """
    agent_loc = state[0:2]
    goal_loc = state[3:5]
    agent_priority = state[5]
    
    dist_goal = distance(start_coord=agent_loc, end_coord=goal_loc, degree=2)
    reward = 1 / dist_goal

    # closest agents detected in all regions around the agent
    closest_base_idx = 6
    priority_base_idx = closest_base_idx + num_regions + 1
    for i in range(num_regions):
        d_xi = state[closest_base_idx + i]
        priority_i = state[priority_base_idx + i]
        reward_i = (agent_priority - priority_i) * (1 - d_xi)
        reward += reward_i
    return reward


def get_reward_old(priorities, goal, sensing_range, curr_state, all_agent_states, robot):
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
    rx = sensing_range
    n = len(priorities) - 1
    pos_x = curr_state
    d_xg = 0  # euclidean distance between pos_x and goal

    reward = 1 / d_xg
    for i in range(n):
        p_i = priorities[i]
        pos_i = all_agent_states[i]
        d_xi = 0  # euclidean distance between pos_x and pos_i

        indicator = 0
        if d_xi < sensing_range:
            indicator = 1

        reward_i = (p_x - p_i) * (1 - d_xi / rx) * indicator
        reward += reward_i

    return reward
