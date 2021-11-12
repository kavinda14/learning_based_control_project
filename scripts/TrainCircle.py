import pickle
import time

from lbc import NeuralNet
from lbc.Grid import Grid
from lbc.Robot import Robot
from lbc.SensorModel import SensorModel, create_binary_matrices
from lbc.Simulator import Simulator

if __name__ == "__main__":
    bounds = [21, 21]

    input_partial_info_binary_matrices = list()
    in_path_mats = list()
    input_actions_binary_matrices = list()
    input_scores = list()

    planner_options = [
        "random",
        # "greedy",
    ]

    input_partial_info_binary_matrices_pickle = open("input_partial_info_binary_matrices_pickle", "wb")
    input_path_matrices_pickle = open("input_path_matrices_pickle", "wb")
    input_actions_binary_matrices_pickle = open("input_actions_binary_matrices_pickle", "wb")
    input_scores_pickle = open("input_scores_pickle", "wb")

    for i in range(45):
        for planner in planner_options:
            start = time.time()
            # Bounds need to be an odd number for the action to always be in the middle
            grid = Grid(bounds, 6, [])

            # Selects random starting locations for the robot
            # We can't use the exact bounds (need -1) due to the limits we create in checking valid location functions
            x_loc, y_loc = grid.random_loc()

            robot = Robot(x_loc, y_loc, bounds, grid)
            sensor_model = SensorModel(robot, grid)

            simulator = Simulator(grid, robot, sensor_model, planner)
            # simulator.visualize()
            simulator.run(2500, False)
            # simulator.visualize()
            # Training data
            path_mats = sensor_model.get_final_path_matrices()

            final_partial_info = sensor_model.get_final_partial_info()
            partial_info_binary_matrices = create_binary_matrices(final_partial_info)

            final_actions = sensor_model.get_final_actions()
            final_actions_binary_matrices = create_binary_matrices(final_actions)

            final_scores = sensor_model.get_final_scores()

            in_path_mats = in_path_mats + path_mats
            input_partial_info_binary_matrices = input_partial_info_binary_matrices + partial_info_binary_matrices
            input_actions_binary_matrices = input_actions_binary_matrices + final_actions_binary_matrices
            input_scores = input_scores + final_scores

            # pickle.dump(path_matricies, input_path_matrices_pickle)
            # pickle.dump(partial_info_binary_matrices, input_partial_info_binary_matrices_pickle)
            # pickle.dump(final_actions_binary_matrices, input_actions_binary_matrices_pickle)
            # pickle.dump(final_scores, input_scores_pickle)

            end = time.time()
            time_taken = (end - start) / 60
            print("Iteration: {}, Planner: {}, Time taken: {:.3f}".format(i, planner, time_taken))

    print("final_path_matrices: ", len(in_path_mats))
    print("final_partial_info_binary_matrices: ", len(input_partial_info_binary_matrices))
    print("final_final_actions_binary_matrices", len(input_actions_binary_matrices))
    print("final_final_scores: ", len(input_scores))

    scores_pickle_in = open("input_scores_pickle", "rb")
    current_score = pickle.load(scores_pickle_in)
    print(len(current_score))

    # ### Train network
    data = NeuralNet.dataset_generator(input_partial_info_binary_matrices, in_path_mats,
                                       input_actions_binary_matrices, input_scores)
    NeuralNet.run_network(data, bounds)
