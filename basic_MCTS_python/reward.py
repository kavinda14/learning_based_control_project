'''
Basic MCTS implementation
Graeme Best
Oregon State University
Jan 2020
'''

from action import Action #, printActionSequence
import random
import SensorModel
import NeuralNet
import torch

# def reward(action_sequence):
    # A simple reward function

    # Iterate through the sequence, looking at pairs
    # reward = 0
    # for i in range(len(action_sequence)-1): # Yes, we want -1 here
        
    #     # Pick out a pair
    #     first = action_sequence[i]
    #     second = action_sequence[i+1]

    #     # Add to the reward if second is +1
    #     if first.id + 1 == second.id:
    #         reward += 1

    # # Also give reward for first action by itself
    # if action_sequence[0].id == 1:
    #     reward += 1

    # # Normalise between 0 and 1
    # max_reward = len(action_sequence) #-1
    # if max_reward == 0:
    #     reward_normalised = 0
    # else:
    #     reward_normalised = float(reward) / float(max_reward)
    # return reward_normalised

def reward(sequence):
    return random.randint(0, 10) + len(sequence)

def greedy_reward(rollout_sequence, sensor_model):
    reward = 0

    for state in rollout_sequence:
        state_loc = state.get_location()
        # print(state_loc)
        # scanned_unobs = sensor_model.scan(state_loc, False)
        reward += len(sensor_model.scan(state_loc, False)[0])
        # reward += len(scanned_unobs[0]) + len(scanned_unobs[1])
        # print('reward: ', reward)

    return reward


# def network_reward(rollout_sequence, sensor_model):

#     model = NeuralNet.Net(map.get_bounds())
#     model.load_state_dict(torch.load("/home/kavi/thesis/neural_net_weights/circles_random_21x21"))
#     # model.load_state_dict(torch.load("/home/kavi/thesis/neural_net_weights/circles_random_21x21_epoch2"))
#     model.eval()

#     partial_info = [sensor_model.create_partial_info(False)]
#     partial_info_binary_matrices = sensor_model.create_binary_matrices(partial_info)

#     path_matrix = sensor_model.create_final_path_matrix(False)

#     final_actions = [sensor_model.create_action_matrix(action, True)]
#     final_actions_binary_matrices = sensor_model.create_binary_matrices(final_actions)

#     input = NeuralNet.create_image(partial_info_binary_matrices, path_matrix, final_actions_binary_matrices)

#     # The unsqueeze adds an extra dimension at index 0 and the .float() is needed otherwise PyTorch will complain
#     # By unsqeezing, we add a batch dimension to the input, which is required by PyTorch: (n_samples, channels, height, width) 
#     input = input.unsqueeze(0).float()

#     action_score = model(input).item()

#     for state in rollout



