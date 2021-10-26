import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

import OraclePlanner

sys.path.insert(0, './basic_MCTS_python')
from basic_MCTS_python import mcts
# from basic_MCTS_python import plot_tree

class Simulator:
    def __init__(self, world_map, robot, sensor_model, planner):
        """
        Inputs:
        world_map: The map to be explored
        robot: the Robot object from Robot.py
        """
        self.map = world_map
        self.robot = robot
        self.sensor_model = sensor_model
        # The following identify what has been seen by the robot
        self.obs_occupied = set()
        self.obs_free = set()
        self.score = 0
        self.iterations = 0

        self.planner = planner
        
        # actions list was created because we thought the robot was sometimes not moving
        self.actions = list()

    def run(self, duration, visualize=False):
        self._update_map()
        self.sensor_model.create_partial_info()
        self.sensor_model.append_score(self.score)
        self.sensor_model.append_path(self.robot.get_loc())
        self.sensor_model.create_final_path_matrix()
        # At the start, there is no action, so we just add the initial partial info into the action matrix list
        initial_partial_info_matrix = self.sensor_model.get_final_partial_info()[0]
        self.sensor_model.append_action_matrix(initial_partial_info_matrix)
        for _ in range(0, duration):
            end = self.tick(visualize)
            if end:
                break

    def tick(self, visualize=False):
        self.iterations += 1

        # Generate an action from the robot path
        # action = OraclePlanner.random_planner(self.robot)
        if self.planner == "random":
            action = OraclePlanner.random_planner(self.robot, self.sensor_model)
        if self.planner == "greedy":
            action = OraclePlanner.greedy_planner(self.robot, self.sensor_model, self.map)
        if self.planner == "network":
            action = OraclePlanner.greedy_planner(self.robot, self.sensor_model, self.map, True)
        if self.planner == 'mcts':
            budget = 7
            max_iterations = 200
            exploration_exploitation_parameter = 0.8 # =1.0 is recommended. <1.0 more exploitation. >1.0 more exploration. 
            solution, root, list_of_all_nodes, winner_node, winner_loc = mcts.mcts(budget, max_iterations, exploration_exploitation_parameter, self.robot, self.sensor_model)
            action = self.robot.get_direction(self.robot.get_loc(), winner_loc)


        self.actions.append(action)
        # print("sensor path: ", self.sensor_model.get_final_path())

        self.sensor_model.create_action_matrix(action)
        # Move the robot
        self.robot.move(action)
        # Update the explored map based on robot position
        self._update_map()
        self.sensor_model.create_partial_info()
        self.sensor_model.append_score(self.score)
        previous_paths_debug = self.sensor_model.get_final_path()
        self.sensor_model.append_path(self.robot.get_loc())
        self.sensor_model.create_final_path_matrix()
        
        if visualize:
            self.visualize()

        # Score is calculated in _update function.
        # It needs to be reset otherwise the score will carry on to the next iteration even if no new obstacles were scanned.
        self.reset_score()

        # End when all objects have been observed OR 1,000 iterations
        if (len(self.obs_occupied) == self.map.unobs_occupied) or (self.iterations == 1000000):
            return True
        else:
            return False

    def reset_game(self):
        self.iterations = 0
        self.score = 0
        self.obs_occupied = set()
        self.obs_free = set()

    def get_score(self):
        return self.score

    def set_score(self, score):
        self.score = score

    def reset_score(self):
        self.score = 0

    def get_iteration(self):
        return self.iterations

    def get_actions(self):
        return self.actions

    def _update_map(self):
        # Sanity check the robot is in bounds
        if not self.robot.check_valid_loc():
            print(self.robot.get_loc())
            raise ValueError(f"Robot has left the map. It is at position: {self.robot.get_loc()}, outside of the map boundary")
        
        new_observations = self.sensor_model.scan(self.robot.get_loc())
        # Score is the number of new obstacles found
        self.set_score(len(new_observations[0]))
        self.obs_occupied = self.obs_occupied.union(new_observations[0])
        self.obs_free = self.obs_free.union(new_observations[1])
        self.map.obs_occupied = self.obs_occupied
        self.map.obs_free = self.obs_free

    def visualize(self):
        plt.xlim(0, self.map.bounds[0])
        plt.ylim(0, self.map.bounds[1])
        plt.title("Planner: {}, Score: {}".format(self.planner, sum(self.sensor_model.get_final_scores())))

        ax = plt.gca()
        ax.set_aspect('equal', 'box')
        
        for spot in self.map.unobs_occupied:
            hole = patches.Rectangle(spot, 1, 1, facecolor='red')
            ax.add_patch(hole)

        for spot in self.map.unobs_free:
            hole = patches.Rectangle(spot, 1, 1, facecolor='black')
            ax.add_patch(hole)
        
        for spot in self.obs_free:
            hole = patches.Rectangle(spot, 1, 1, facecolor='white')
            ax.add_patch(hole)
        
        for spot in self.obs_occupied:
            hole = patches.Rectangle(spot, 1, 1, facecolor='green')
            ax.add_patch(hole)

        # Plot robot
        robot_x = self.robot.get_loc()[0] + 0.5
        robot_y = self.robot.get_loc()[1] + 0.5
        plt.scatter(robot_x, robot_y, color='purple', zorder=5)

        # Plot robot path
        x_values = list()
        y_values = list()
        for path in self.sensor_model.get_final_path():
            x_values.append(path[0] + 0.5)
            y_values.append(path[1] + 0.5)

        plt.plot(x_values, y_values)


        plt.show()