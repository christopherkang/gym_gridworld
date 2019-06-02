import numpy as np
import pickle as p


class Gridworld:
    def __init__(self, worldSize, action_specs, parameters):
        """Create Gridworld.

        The indices go from 0..xSize - 1 and 0..ySize - 1

        Arguments:
            xSize {int} -- size of matrix in axis 1
            ySize {int} -- size of matrix in axis 0
            objects {matrix; n x 3} -- objects to be placed on grid
        """
        self.x_size = worldSize[0]
        self.y_size = worldSize[1]
        self.representation = np.zeros(worldSize)
        self.blocks = np.zeros(worldSize)
        self.collision_penalty = parameters[1]

        # form: [reward, passable (bool), xCoord, yCoord]
        self.item_list = np.array(parameters[0])
        self.epoch = 0
        self.ACTION_BANK = action_specs[0]
        self.ACTION_EFFECTS = action_specs[1]
        # reversed because its rows x columns
        self.does_agent_exist = False
        for [value, canEnter, x_coord, y_coord] in parameters[0]:
            self.representation[2, 2] = 0.1
            self.representation[int(y_coord), int(x_coord)] = value
            if (not canEnter):
                self.blocks[y_coord, x_coord] = 1

    def __str__(self):
        return self.get_representation(True, True)

    def place_agent(self, x_coord, y_coord):
        """Place agent on map; assumes agent doesn't exist.

        Arguments:
            xCoord {int} -- coordinate of x position
            yCoord {int} -- coordinate of y position

        Returns:
            bool -- returns False if agent exist

        """
        if (self.does_agent_exist):
            return False
        else:
            self.x_agent = x_coord
            self.y_agent = y_coord
            self.does_agent_exist = True
            self.prox_map = self.calculate_prox_map((0, 0))

    def distance_to_objects(self, x_coord, y_coord):
        """Return matrix with distance to relevant objects

        Arguments:
            xCoord {int} -- xCoord to compare object to
            yCoord {int} -- yCoord to compare object to

        Returns:
            matrix -- matrix of each item and it's value / x delta / y delta
        """

        distance_matrix = [[row[0], row[1] - x_coord, row[2] - y_coord]
                           for row in self.item_list if row[1]]
        return distance_matrix

    def get_agent_coords(self):
        """Return agent coordinates (assumes agent has been placed).

        Returns:
            (int, int) -- coordinates of agent

        """
        return (self.x_agent, self.y_agent)

    def move_possible(self, xy_tuple):
        """Identify if a move is possible. Requires agent to be initialized

        Arguments:
            xy_tuple {Tuple} -- change in x and y position

        Returns:
            Bool -- returns whether the move is possible
        """

        xEnd = self.x_agent + xy_tuple[0]
        yEnd = self.y_agent + xy_tuple[1]
        if (xEnd < 0 or xEnd > self.x_size - 1 or
            yEnd < 0 or yEnd > self.y_size - 1 or
                self.blocks[yEnd, xEnd] == 1):
            return False
        return True

    def move_agent(self, action):
        """Move agent given string action.

        If the agent would go off the grid, the agent maintains its current
            position

        Arguments:
            action {int} -- argument for how to move. refers to an index in a
                common action index bank

        Returns:
            int -- Returns reward after action

        """
        if (self.does_agent_exist):
            self.update_proximity_map(self.ACTION_EFFECTS[action])
            return self.appropriate_move(self.ACTION_EFFECTS[action])
        else:
            raise Exception("Agent does not exist!")

    def appropriate_move(self, xy_tuple, debugging=False):
        """Decide if a move is appropriate and take it if necessary

        Arguments:
            xy_tuple {Tuple} -- change in x and y position

        Returns:
            int -- return reward of moving to that square
        """

        xEnd = self.x_agent + xy_tuple[0]
        yEnd = self.y_agent + xy_tuple[1]
        if debugging:
            print(f"Target end position: ({xEnd}, {yEnd})")
        if (self.move_possible(xy_tuple)):
            self.x_agent = xEnd
            self.y_agent = yEnd
            self.epoch += 1
            output = self.representation[self.y_agent, self.x_agent]

            # clear item
            self.representation[self.y_agent, self.x_agent] = 0

            # remove item from proximity map
            for row in self.item_list:
                if (row[2] == self.x_agent and row[3] == self.y_agent):
                    row[0] = 0
            return output
        return self.collision_penalty

    def get_representation(self, showAgent=False, scaleEnvironment=False,
                           cherry_color="FF0000", bomb_color="0000FF"):
        """Return matrix form of total environment.

        Keyword Arguments:
            scaleEnvironment {bool} -- whether to represent the rewards in a
                scaled way to allow better visual representation
            showAgent {bool} -- whether to show the agent in the
                matrix representation (default: {False})

        Returns:
            matrix -- matrix representing the environment

        """

        def int_array_color(color_string):
            return np.array([int(color_string[0:2], 16), int(
                color_string[2:4], 16), int(color_string[4:], 16)])

        def replace_color(object_at_location):
            # print(f"objaloc:{object_at_location}")
            if object_at_location[0] == 1:
                # input("no i'ave happened cherry")
                return int_array_color(cherry_color) / 255
            elif object_at_location[0] == -1:
                # input("I'VE HAPPENED????")
                return int_array_color(bomb_color) / 255
            else:
                return np.array([0, 0, 0])

        copy = np.expand_dims(self.representation, axis=2)  # make a copy
        output = np.concatenate(
            (copy, np.zeros((copy.shape[0], copy.shape[1], 2))), axis=2)
        # output = np.swapaxes(output, 0, 2)
        for row in output:
            # print(f"row:{row}")
            for element in row:
                element = replace_color(element)
        # output = np.swapaxes(output, 0, 2)
        # if (scaleEnvironment):
        #     output /= max(output.max(), 1) * 2
        if (showAgent):
            output[self.y_agent, self.x_agent] = [1, 1, 1]
        # output[2, 2] = int_array_color(cherry_color)
        # print(f"output: {output}")
        return output

    def return_vision(self, x_dist, y_dist):
        """Return what an agent "would" be able to see

        Arguments:
            xDist {int} -- distance the agent can see in the x direction
            yDist {int} -- distance the agent can see in the y direction

        Returns:
            matrix or boolean -- matrix of the vision with the reward values,
            false if there is an unexpected error
        """

        if (self.does_agent_exist):
            output = np.zeros((2 * y_dist + 1, 2 * x_dist + 1))

            for x_coord in range(-x_dist, x_dist + 1):
                for y_coord in range(-y_dist, y_dist + 1):
                    x_pos = self.x_agent + x_coord
                    y_pos = self.y_agent + y_coord

                    if (x_pos < 0 or x_pos >= self.x_size) or (
                            y_pos < 0 or y_pos >= self.y_size):
                        output[y_coord + y_dist, x_coord + x_dist] = -np.infty
                    else:
                        output[y_coord + y_dist, x_coord +
                               x_dist] = self.representation[y_pos, x_pos]
            return output
        else:
            return False

    def simulate_action(self, xy_tuple):
        """Return simulated matrix if a specific move was made

        Arguments:
            xy_tuple {Tuple} -- change in x and y position

        Returns:
            matrix -- matrix representation of environment
        """

        output = np.array(self.get_representation(True, True))
        if (self.move_possible(xy_tuple)):
            output[self.y_agent, self.x_agent] = 0
            output[self.y_agent + xy_tuple[1], self.x_agent + xy_tuple[0]] = 1
            return output
        else:
            return output

    def get_epoch(self):
        """Return the epoch of the current trial

        Returns:
            int -- epoch of the world
        """

        return self.epoch

    def update_proximity_map(self, xy_tuple, speculative=False):
        """Update proximity map

        Arguments:
            xy_tuple {Tuple} -- change in x and y position

        Keyword Arguments:
            speculative {bool} -- Choose whether to update the proximity map
                or speculate the action and output the would-be result
                (default: {False})

        Returns:
            matrix -- matrix of the map (only if speculative True)
        """

        if speculative:
            return self.calculate_prox_map(xy_tuple)
        else:
            self.prox_map = self.calculate_prox_map(xy_tuple)

    def calculate_prox_map(self, xy_tuple):
        """Calculate the proximity map

        Returns:
            matrix -- format of [Reward, x distance, y distance]
        """

        prox_map = np.array([row[[0, 2, 3]]
                             for row in self.item_list if row[1]], dtype=np.float, copy=True)
        # np.reshape(prox_map, (-1, 3))
        prox_map[:, 1] -= (np.clip(self.x_agent +
                                   xy_tuple[0], 0, self.x_size - 1))
        prox_map[:, 2] -= (np.clip(self.y_agent +
                                   xy_tuple[1], 0, self.y_size - 1))

        # scaling
        prox_map[:, 1] = prox_map[:, 1] / float(self.x_size - 1)
        prox_map[:, 2] = prox_map[:, 2] / float(self.y_size - 1)

        # make hit items worth 0
        for row in prox_map:
            if row[1] == 0 and row[2] == 0:
                row[0] = 0

        return prox_map

    def calculate_grid_map(self, xy_tuple):
        # we are going to set the first col to be the agent
        out_map_x = np.zeros(
            (len(self.item_list) + 1, len(self.item_list) + 1))
        out_map_y = np.zeros(
            (len(self.item_list) + 1, len(self.item_list) + 1))

        new_agent_x = np.clip(self.x_agent + xy_tuple[0], 0, self.x_size - 1)
        new_agent_y = np.clip(self.y_agent + xy_tuple[1], 0, self.y_size - 1)

        out_map_x[0][0] = new_agent_x
        out_map_y[0][0] = new_agent_y

        # assign the values on the diagnols
        for index in range(1, len(self.item_list) + 1):
            out_map_x[index][index] = self.item_list[index - 1][2]
            out_map_y[index][index] = self.item_list[index - 1][3]

        # we will first compute distances from the agent
        for target in range(len(self.item_list)):
            out_map_x[target + 1][0] = new_agent_x - self.item_list[target][2]
            out_map_y[target + 1][0] = new_agent_y - self.item_list[target][3]

        # now, we will compute distances from objects to each other
        for source in range(1, len((self.item_list)) + 1):
            for target in range(source + 1, len(self.item_list) + 1):
                distance = self.item_list[source - 1][2:] - \
                    self.item_list[target - 1][2:]
                out_map_x[target][source] = distance[0]
                out_map_x[target][source] = distance[1]

        # print(out_map_x, out_map_y)

        return out_map_x, out_map_y

    def load_world(self, directory):
        """Load world from directory

        Arguments:
            directory {string} -- string of directory, including the name of the pickle file
        """

        parameters = p.load(open(directory, 'rb'))
        self.item_list = parameters[0]
        self.collision_penalty = parameters[1]

