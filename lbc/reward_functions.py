"""
Reward functions for lbc agent.
"""
import numpy as np


def prio_reward(state, action: list):
    """
    Gets the reward for a specific state-action pair based on priorities and distance to goal.
    State: [pos x, pos y, distance to goal, priority, 8 * normalized distance to closest agent, 8 * priorities]

    :param state:       State encoding of the agent
                            [0,1] location of agent
                            [2] priority of agent
                            [3,4] goal location
                            Rest of the array contains information regarding the surroundings of the agent
                            The first part is the distances to the closest agent in each region
                            The second part if the priority of the closest agent in each region
                            These two parts are the same (variable_ length
    :param action:      Action taken by the agent
                            Agent is able to take a single movement in a vector around it
    :return:            reward for this state
    """
    agent_loc = state[0:2]
    agent_priority = state[2]
    goal_loc = state[3:5]

    # todo: use diagonal from board_size to normalize instead of hardcoding
    dist_goal = np.linalg.norm(goal_loc - agent_loc) / np.sqrt(200)  # normalized distance to goal relative to diagonal
    reward = 1 / (dist_goal + 10**-6)

    # closest agents detected in all regions around the agent
    region_info = state[5:]
    mid_idx = len(region_info)//2
    dist_info = region_info[:mid_idx]
    prio_info = region_info[mid_idx:]
    for each_dist, each_prio in zip(dist_info, prio_info):
        reward_i = (agent_priority - each_prio) * (1 - each_dist)
        reward += reward_i
    return reward


def goal_reward(state, action: list):
    """
    Gets the reward for a specific state-action pair based only on distance to goal.
    State: [pos x, pos y, distance to goal, priority, 8 * normalized distance to closest agent, 8 * priorities]

    :param state:       State encoding of the agent
                            [0,1] location of agent
                            [2] priority of agent
                            [3,4] goal location
                            Rest of the array contains information regarding the surroundings of the agent
                            The first part is the distances to the closest agent in each region
                            The second part if the priority of the closest agent in each region
                            These two parts are the same (variable_ length
    :param action:      Action taken by the agent
                            Agent is able to take a single movement in a vector around it
    :return:            reward for this state
    """
    agent_loc = state[0:2]
    goal_loc = state[3:5]

    # todo: use diagonal from board_size to normalize instead of hardcoding
    dist_goal = np.linalg.norm(goal_loc - agent_loc) / np.sqrt(200)  # normalized distance to goal relative to diagonal
    reward = 1 / (dist_goal + 10**-6)
    return reward
