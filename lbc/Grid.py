import numpy as np

from lbc.utils import euclidean_distance


class Map:
    def __init__(self, bounds, num_obstacles, unobs_occupied_set, given=False):
        """
        Inputs:
            robot: list of robot objects from Robot.py
        """
        # NOTE: These values scale the difficulty of the problem
        self.num_obstacles = num_obstacles
        self.bounds = bounds

        # The first two were kept as lists because I am iterating and modifying
        # it at the same time in the scan function.
        self.unobs_occupied = set()
        self.unobs_free = set()

        # These two are only populated in the Simulator.
        self.obs_occupied = set()
        self.obs_free = set()

        # Adds squares
        if not given:
            # Add obstacle to middle of map
            x = self.bounds[0] // 2
            y = self.bounds[1] // 2
            mid_point = (x, y)
            self.unobs_occupied.add(mid_point)

            for x in range(bounds[0]):
                for y in range(bounds[1]):
                    distance = euclidean_distance((x, y), mid_point)
                    if distance < 1.6:  # 1.6 with map of 21, 21 bounds is good for square
                        # if distance < 2.3: # 4.3 with map of 41, 41 bounds is good for circle
                        self.unobs_occupied.add((x, y))

            # Add obstacles to environment
            for i in range(self.num_obstacles):
                x = int(np.random.uniform(3, self.bounds[0] - 2, size=1))
                y = int(np.random.uniform(3, self.bounds[1] - 2, size=1))
                mid_point = (x, y)
                if mid_point not in self.unobs_occupied:
                    self.unobs_occupied.add(mid_point)
                    # self.unobs_free.remove(mid_point)
                    for x in range(bounds[0]):
                        for y in range(bounds[1]):
                            # for free_loc in list(self.unobs_free):
                            distance = euclidean_distance((x, y), mid_point)
                            if distance < 1.6:  # 1.6 with map of 21, 21 bounds is good for square
                                # if distance < 2.3: # 4.3 with map of 41, 41 bounds is good for circle
                                self.unobs_occupied.add((x, y))
        else:
            self.unobs_occupied = unobs_occupied_set

        # Add free coords to unobs_free list
        for x in range(bounds[0]):
            for y in range(bounds[1]):
                if (x, y) not in self.unobs_occupied:
                    self.unobs_free.add((x, y))

        self.reset_unobs_free = set(self.unobs_free)
        self.reset_unobs_occupied = set(self.unobs_occupied)

    # This was written to select random starting locations for training
    def check_loc(self, x_loc, y_loc):
        x = x_loc
        y = y_loc
        in_bounds = (0 <= x < self.bounds[0] and 0 <= y < self.bounds[1])

        # Check unobs_occupied and obs_occupied from map
        for loc in self.unobs_occupied:
            if x == loc[0] and y == loc[1]:
                return False

        for loc in self.obs_occupied:
            if x == loc[0] and y == loc[1]:
                return False

        return in_bounds

    # Function is used in testing because we need to keep the same map to make it fair
    def reset_map(self):
        self.obs_occupied = set()
        self.obs_free = set()

    def get_unobs_free(self):
        return self.unobs_free

    def get_unobs_occupied(self):
        return self.unobs_occupied

    def set_unobs_free(self, unobs_free_set):
        self.unobs_free = unobs_free_set

    def set_unobs_occupied(self, unobs_occupied_set):
        self.unobs_occupied = unobs_occupied_set

    def get_bounds(self):
        return self.bounds
