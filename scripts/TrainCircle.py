from lbc.SensorModel import SensorModel, create_binary_matrices
from lbc.Grid import Grid
from lbc.Robot import Robot
from lbc.Simulator import Simulator
from lbc import NeuralNet
import random
import time

if __name__ == "__main__":
    input_partial_info_binary_matrices = []
    input_path_matrices = []
    input_actions_binary_matrices = []
    input_scores = []

    planner_options = ["random", "greedy"]
    # planner_options = ["greedy"]
    
    for i in range(45):
        for planner in planner_options: 
            start = time.time()
            # Bounds need to be an odd number for the action to always be in the middle
            bounds = [21, 21]
            grid = Grid(bounds, 6, [])

            # Selects random starting locations for the robot
            # We can't use the exact bounds (need -1) due to the limits we create in checking valid location functions
            valid_starting_loc = False
            while not valid_starting_loc:
                x = random.randint(0, bounds[0]-1)
                y = random.randint(0, bounds[0]-1)
                valid_starting_loc = grid.check_loc(x, y)

            robot = Robot(x, y, bounds, grid)
            sensor_model = SensorModel(robot, grid)
            
            simulator = Simulator(grid, robot, sensor_model, planner)
            # simulator.visualize()
            simulator.run(2500, False)

            # simulator.visualize()
            
            # Training data
            path_matrices = sensor_model.get_final_path_matrices()

            final_partial_info = sensor_model.get_final_partial_info()
            partial_info_binary_matrices = create_binary_matrices(final_partial_info)

            final_actions = sensor_model.get_final_actions()
            final_actions_binary_matrices = create_binary_matrices(final_actions)

            final_scores = sensor_model.get_final_scores()

            input_path_matrices = input_path_matrices + path_matrices
            input_partial_info_binary_matrices = input_partial_info_binary_matrices + partial_info_binary_matrices
            input_actions_binary_matrices = input_actions_binary_matrices + final_actions_binary_matrices
            input_scores = input_scores + final_scores

            end = time.time()
            time_taken = (end - start)/60
            print("Iteration: {}, Planner: {}, Time taken: {:.3f}".format(i, planner, time_taken))

    print("final_path_matrices: ", len(input_path_matrices))
    print("final_partial_info_binary_matrices: ", len(input_partial_info_binary_matrices))
    print("final_final_actions_binary_matrices", len(input_actions_binary_matrices))
    print("final_final_scores: ", len(input_scores))
    
    # ### Train network
    data = NeuralNet.dataset_generator(input_partial_info_binary_matrices, input_path_matrices, input_actions_binary_matrices, input_scores)
    NeuralNet.run_network(data, bounds)
