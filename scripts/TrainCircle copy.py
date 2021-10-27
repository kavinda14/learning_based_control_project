import pickle
import random
import time

from lbc.Grid import Map
from lbc.Robot import Robot
from lbc.SensorModel import SensorModel
from lbc.Simulator import Simulator

if __name__ == "__main__":

    input_partial_info_binary_matrices = list()
    input_path_matrices = list()
    input_actions_binary_matrices = list()
    input_scores = list()

    # planner_options = ["random", "greedy"]
    planner_options = ["random"]

    input_partial_info_binary_matrices_pickle = open("input_partial_info_binary_matrices_pickle", "wb")
    input_path_matrices_pickle = open("input_path_matrices_pickle", "wb")
    input_actions_binary_matrices_pickle = open("input_actions_binary_matrices_pickle", "wb")
    input_scores_pickle = open("input_scores_pickle", "wb")
    
    limit = 4
    for i in range(limit):
        for planner in planner_options: 
            start = time.time()
            # Bounds need to be an odd number for the action to always be in the middle
            bounds = [41, 41]
            map = Map(bounds, 7, [])

            # Selects random starting locations for the robot
            # We can't use the exact bounds (need -1) due to the limits we create in checking valid location functions
            valid_starting_loc = False
            while not valid_starting_loc:
                x = random.randint(0, bounds[0]-1)
                y = random.randint(0, bounds[0]-1)
                valid_starting_loc = map.check_loc(x, y) 

            robot = Robot(x, y, bounds, map)
            sensor_model = SensorModel(robot, map)
            
            simulator = Simulator(map, robot, sensor_model, planner)
            # simulator.run(10000, False)
            simulator.run(20, False)

            # simulator.visualize()
            
            ### Training data
            path_matricies = sensor_model.get_final_path_matrices()

            final_partial_info = sensor_model.get_final_partial_info()
            partial_info_binary_matrices = create_binary_matrices(final_partial_info)

            final_actions = sensor_model.get_final_actions()
            final_actions_binary_matrices = create_binary_matrices(final_actions)

            final_scores = sensor_model.get_final_scores()

            pickle.dump(path_matricies, input_path_matrices_pickle)
            pickle.dump(partial_info_binary_matrices, input_partial_info_binary_matrices_pickle)
            pickle.dump(final_actions_binary_matrices, input_actions_binary_matrices_pickle)
            pickle.dump(final_scores, input_scores_pickle)
            # input_path_matrices = input_path_matrices + path_matricies
            # input_partial_info_binary_matrices = input_partial_info_binary_matrices + partial_info_binary_matrices
            # input_actions_binary_matrices = input_actions_binary_matrices + final_actions_binary_matrices
            # input_scores = input_scores + final_scores

            end = time.time()
            time_taken = (end - start)/60
            print("Iteration: {}, Planner: {}, Time taken: {:.3f}".format(i, planner, time_taken))

    # print("final_path_matrices: ", len(input_path_matrices))
    # print("final_partial_info_binary_matrices: ", len(input_partial_info_binary_matrices))
    # print("final_final_actions_binary_matrices", len(input_actions_binary_matrices))
    # print("final_final_scores: ", len(input_scores))

    scores_pickle_in = open("input_scores_pickle", "rb")
    current_score = pickle.load(scores_pickle_in)
    print(len(current_score))
    
    ### Train network
    # data = NeuralNet.datasetGenerator(limit, "input_partial_info_binary_matrices_pickle", "input_path_matrices_pickle", "input_actions_binary_matrices_pickle", "input_scores_pickle")
    # NeuralNet.runNetwork(data, bounds)