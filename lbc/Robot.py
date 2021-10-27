class Robot:
    def __init__(self, x, y, bounds, map):
        # Variables that changes
        self.x_loc = x
        self.y_loc = y
        # Variable to track where the robot has been
        self.path = list()
        self.index = 0

        # Static variables
        self.start_loc = self.x_loc, self.y_loc
        self.velocity = 1.0
        self.sensing_range = 2.85  # Range for square with bounds 21, 21
        # self.sensing_range = 4.3 # Range for circles with bounds 41, 41
        self.lim = bounds
        self.map = map

    def reset_robot(self):
        self.x_loc = self.start_loc[0]
        self.y_loc = self.start_loc[1]
        self.path = list()
        self.index = 0

    def check_valid_loc(self):
        x = self.x_loc
        y = self.y_loc
        in_bounds = (0 <= x <= self.lim[0] and 0 <= y <= self.lim[1])

        # Check if new location is intersecting with obstacles from map
        for loc in self.map.unobs_occupied:
            if x == loc[0] and y == loc[1]:
                print("Invalid Location {}".format(loc))
                return False

        for loc in self.map.obs_occupied:
            if x == loc[0] and y == loc[1]:
                print("Invalid Location {}".format(loc))
                return False

        return in_bounds

    def check_new_loc(self, x_loc, y_loc):
        x = x_loc
        y = y_loc
        in_bounds = (0 <= x < self.lim[0] and 0 <= y < self.lim[1])

        # Check unobs_occupied and obs_occupied from map
        for loc in self.map.unobs_occupied:
            if x == loc[0] and y == loc[1]:
                return False

        for loc in self.map.obs_occupied:
            if x == loc[0] and y == loc[1]:
                return False

        return in_bounds

    def get_loc(self):
        return self.x_loc, self.y_loc

    def set_loc(self, x_loc, y_loc):
        self.x_loc = x_loc
        self.y_loc = y_loc

    def check_valid_move(self, direction, update_state=False):
        """ Checks if the direction is valid
        direction (str): "left", "right", "up", "down" directions to move the robot
        updateState (bool): if True, function also moves the robot if direction is valid
                            otherwise, only perform validity check without moving robot
        """
        # Just don't move
        if not direction:
            return True

        if direction == 'left':
            valid = self.check_new_loc(self.x_loc - 1, self.y_loc)
            if valid and update_state:
                self.x_loc -= 1

        elif direction == 'right':
            valid = self.check_new_loc(self.x_loc + 1, self.y_loc)
            if valid and update_state:
                self.x_loc += 1

        elif direction == 'backward':
            valid = self.check_new_loc(self.x_loc, self.y_loc + 1)
            if valid and update_state:
                self.y_loc += 1

        elif direction == 'forward':
            valid = self.check_new_loc(self.x_loc, self.y_loc - 1)
            if valid and update_state:
                self.y_loc -= 1
        else:
            raise ValueError(f"Robot received invalid direction: {direction}!")

        return valid

    def check_valid_move_mcts(self, direction, location, updateState=False):
        """ Checks if the direction is valid
        direction (str): "left", "right", "up", "down" directions to move the robot
        updateState (bool): if True, function also moves the robot if direction is valid
                            otherwise, only perform validity check without moving robot
        """
        # Just don't move
        if not direction:
            return True

        x_loc = location[0]
        y_loc = location[1]

        if direction == 'left':
            valid = self.check_new_loc(x_loc - 1, y_loc)
            if valid and updateState:
                x_loc -= 1

        elif direction == 'right':
            valid = self.check_new_loc(x_loc + 1, y_loc)
            if valid and updateState:
                x_loc += 1

        elif direction == 'backward':
            valid = self.check_new_loc(x_loc, y_loc + 1)
            if valid and updateState:
                y_loc += 1

        elif direction == 'forward':
            valid = self.check_new_loc(x_loc, y_loc - 1)
            if valid and updateState:
                y_loc -= 1
        else:
            raise ValueError(f"Robot received invalid direction: {direction}!")

        if updateState:
            return [valid, [x_loc, y_loc]]

        return valid

    def move(self, direction):
        """ Move the robot while respecting bounds"""
        self.check_valid_move(direction, update_state=True)

    def get_action_loc(self, action):
        robot_loc = self.get_loc()
        action_loc = []

        if action == 'left':
            action_loc = [robot_loc[0] - 1, robot_loc[1]]

        elif action == 'right':
            action_loc = [robot_loc[0] + 1, robot_loc[1]]

        elif action == 'backward':
            action_loc = [robot_loc[0], robot_loc[1] + 1]

        elif action == 'forward':
            action_loc = [robot_loc[0], robot_loc[1] - 1]

        return action_loc

    @staticmethod
    def get_direction(current_loc, next_loc):
        if next_loc[0] - current_loc[0] == -1:
            return 'left'
        if next_loc[0] - current_loc[0] == 1:
            return 'right'
        if next_loc[1] - current_loc[1] == 1:
            return 'backward'
        if next_loc[1] - current_loc[1] == -1:
            return 'forward'

        return None

    def get_bounds(self):
        return self.lim
