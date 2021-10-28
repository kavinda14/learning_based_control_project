import copy
import random as r
import time as time

from lbc.Grid import Map
from lbc.Robot import Robot
from lbc.SensorModel import SensorModel
from lbc.Simulator import Simulator

if __name__ == "__main__":
 
    # Bounds need to be an odd number for the action to always be in the middle
    # planner_options = ["random", "greedy", "network", "mcts"]
    planner_options = ["mcts"]
    # planner_options = ["network"]
    # bounds = [21, 21]
    bounds = [21, 21]
    random = list()
    greedy = list()
    network = list()
    x1 = list()

    # trials = 200
    trials = 1
    for i in range(trials):
        print("Trial no: {}".format(i))
        x1.append(i)
        map = Map(bounds, 7, (), False)
        unobs_occupied = copy.deepcopy(map.get_unobs_occupied())

        valid_starting_loc = False
        while not valid_starting_loc:
            x = r.randint(0, bounds[0]-1)
            y = r.randint(0, bounds[0]-1)
            valid_starting_loc = map.check_loc(x, y) 
        for planner in planner_options:     
            map = Map(bounds, 18, copy.deepcopy(unobs_occupied), True)
            robot = Robot(x, y, bounds, map)
            sensor_model = SensorModel(robot, map)
            start = time.time()
            simulator = Simulator(map, robot, sensor_model, planner)
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

    # plt.plot(x1, random, label = "random")
    # plt.plot(x1, greedy, label = "greedy")
    # plt.plot(x1, network, label = "network")

    # plt.xlabel('Trial no')
    # # Set the y axis label of the current axis.
    # plt.ylabel('Score')
    # # Set a title of the current axes.
    # plt.title('Avg scores: random: {}, greedy: {}, network: {}'.format(avg_random, avg_greedy, avg_network))
    # # show a legend on the plot
    # plt.legend()
    # # Display a figure.
    # plt.show()


