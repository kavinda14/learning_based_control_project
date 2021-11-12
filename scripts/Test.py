import copy
import time as time

from lbc.Grid import Grid
from lbc.Robot import Robot
from lbc.SensorModel import SensorModel
from lbc.Simulator import Simulator

if __name__ == "__main__":
    # Bounds need to be an odd number for the action to always be in the middle
    planner_options = [
        "random",
        # "greedy",
        # "network",
        # "mcts"
    ]
    bounds = [21, 21]
    max_trials = 200
    random = list()
    greedy = list()
    network = list()
    x1 = list()

    for trial_idx in range(max_trials):
        print("Trial no: {}".format(trial_idx))
        x1.append(trial_idx)
        grid = Grid(bounds, 7, (), False)
        unobs_occupied = copy.deepcopy(grid.unobs_occupied)

        x_start, y_start = grid.random_loc()
        for planner in planner_options:
            grid = Grid(bounds, 18, copy.deepcopy(unobs_occupied), True)
            robot = Robot(x_start, y_start, bounds, grid)
            sensor_model = SensorModel(robot, grid)
            start = time.time()
            simulator = Simulator(grid, robot, sensor_model, planner)
            simulator.visualize()
            simulator.run(50, False)
            end = time.time()
            simulator.visualize()
            score = sum(sensor_model.get_final_scores())
            
    #         print("Planner: {}, Score: {}".format(planner, score))
    #         print("No of steps taken: ", len(simulator.get_actions()))
    #         print("Time taken: ", end - start)

    #         if planner == "random":
    #             random.append(score)
    #         elif planner == "greedy":
    #             greedy.append(score)
    #         else:
    #             network.append(score)

    # avg_random = sum(random)/trials
    # avg_greedy = sum(greedy)/trials
    # avg_network = sum(network)/trials
