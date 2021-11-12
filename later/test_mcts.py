import copy
import random as r

from lbc.Grid import Grid
from lbc.Robot import Robot
from lbc.SensorModel import SensorModel
from lbc.Simulator import Simulator
from lbc.mcts import mcts

if __name__ == "__main__":
    # Bounds need to be an odd number for the action to always be in the middle
    bounds = [21, 21]

    grid = Grid(bounds, 7, (), False)
    unobs_occupied = copy.deepcopy(grid.unobs_occupied)

    valid_starting_loc = False
    while not valid_starting_loc:
        x = r.randint(0, bounds[0] - 1)
        y = r.randint(0, bounds[0] - 1)
        valid_starting_loc = grid.check_loc(x, y)

    grid = Grid(bounds, 18, copy.deepcopy(unobs_occupied), True)
    starting_loc = grid.random_loc()
    robot = Robot(starting_loc[0], starting_loc[1], bounds, grid)
    sensor_model = SensorModel(robot, grid)
    simulator = Simulator(grid, robot, sensor_model, 'greedy')
    # simulator.visualize()

    # Setup the problem
    # num_actions = 3
    # action_set = []
    # for i in range(num_actions):
    #     id = i
    #     action_set.append(Action(id,i))
    budget = 7

    exploration_exploitation_parameter = 0.8  # =1.0 is recommended. <1.0 more exploitation. >1.0 more exploration.
    max_iterations = 20

    print('robot loc: ', starting_loc)
    robot_move = -1
    for i in range(100):
        current_loc = robot.get_loc()
        # print('current_loc: ', current_loc)
        [solution, root, list_of_all_nodes, winner_node, winner_loc] = mcts.mcts(budget, max_iterations,
                                                                                 exploration_exploitation_parameter,
                                                                                 robot, sensor_model)
        next_loc = winner_node.get_coords()
        # print('next_loc: ', next_loc)
        direction = robot.get_direction(current_loc, next_loc)
        # print(direction)
        robot.move(direction)
        # simulator.visualize()

    # simulator.visualize()

    # print('winner: ', winner.get_coords())
