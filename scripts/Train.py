import time

from lbc import NeuralNet
from lbc.Grid import Grid
from lbc.Robot import Robot
from lbc.SensorModel import SensorModel
from lbc.Simulator import Simulator

if __name__ == "__main__":
    bounds = [21, 21]

    input_partial_info_binary_matrices = list()
    input_path_matrices = list()
    input_actions_binary_matrices = list()
    input_scores = list()

    # planner_options = ["random", "greedy"]
    planner_options = ["random"]
    for i in range(50):
        for planner in planner_options: 
            start = time.time()
            # Bounds need to be an odd number for the action to always be in the middle
            grid = Grid(bounds, 18, [])

            # Selects random starting locations for the robot
            # We can't use 111 due to the limits we create in checking valid location functions
            starting_loc = grid.random_loc()

            robot = Robot(starting_loc[0], starting_loc[1], bounds, grid)
            sensor_model = SensorModel(robot, grid)
            
            simulator = Simulator(grid, robot, sensor_model, planner)
            simulator.run(5000, False)

            # simulator.visualize()
            
            ### Training data
            path_matricies = sensor_model.get_final_path_matrices()

            final_partial_info = sensor_model.get_final_partial_info()
            partial_info_binary_matrices = create_binary_matrices(final_partial_info)

            final_actions = sensor_model.get_final_actions()
            final_actions_binary_matrices = create_binary_matrices(final_actions)

            final_scores = sensor_model.get_final_scores()

            input_path_matrices = input_path_matrices + path_matricies
            input_partial_info_binary_matrices = input_partial_info_binary_matrices + partial_info_binary_matrices
            input_actions_binary_matrices = input_actions_binary_matrices + final_actions_binary_matrices
            input_scores = input_scores + final_scores
            print(input_scores)

            end = time.time()
            time_taken = (end - start)/60
            print("Iteration: {}, Planner: {}, Time taken: {:.3f}".format(i, planner, time_taken))

    print("final_path_matrices: ", len(input_path_matrices))
    print("final_partial_info_binary_matrices: ", len(input_partial_info_binary_matrices))
    print("final_final_actions_binary_matrices", len(input_actions_binary_matrices))
    print("final_final_scores: ", len(input_scores))
    
    ### Train network
    data = NeuralNet.dataset_generator(input_partial_info_binary_matrices, input_path_matrices, input_actions_binary_matrices, input_scores)
    NeuralNet.run_network(data, bounds)



