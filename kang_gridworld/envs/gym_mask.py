import sys

import cv2
import gym
import numpy as np

from .gridworld import Gridworld
from gym import spaces

# from gym import error, spaces, utils
# from gym.utils import seeding


class KangGrid(gym.Env):
    metadata = {'render.modes': ['human']}
    _ACTION_BANK = ["UP", "LEFT", "DOWN", "RIGHT"]
    _ACTION_DEF = [[0, -1],
                   [-1, 0],
                   [0, 1],
                   [1, 0]]
    _ACTION_INFO = [_ACTION_BANK, _ACTION_DEF]
    _RENDER_TIME = 1500  # time to render, in milliseconds

    def _get_objects(self):
        """Wrapper for get_objects method of the gridworld

        Returns:
            list -- Returns an array of objects. See gridworld implementation
            for details
        """
        return self.env._get_objects()

    def _set_render_time(self, time):
        self._RENDER_TIME = time

    def _randomly_create_objects(self, number_of_objects, xyDimension,
                                 reward=None, reward_map=None):
        """Creates random objects matching 'parameters' format

        Arguments:
            number_of_objects {int} -- number of objects to be generated
            reward {int} -- reward of individual object
            xyDimension {tuple} -- dimension of world

        Keyword Arguments:
            reward_map {array of int} -- list of rewards to use (default: {""})

        Returns:
            list -- list of objects
        """
        xSize = xyDimension[0]
        ySize = xyDimension[1]

        coordinates = np.random.choice(
            xSize * ySize - 1, number_of_objects, replace=False) + 1

        output = []

        if reward_map:
            for i in range(number_of_objects):
                toAdd = [reward_map[i], True, int(coordinates[i] %
                                                  xSize), int(coordinates[i] // xSize)]
                output.append(toAdd)
        else:
            for i in range(number_of_objects):
                toAdd = [reward, True, int(coordinates[i] %
                                           xSize), int(coordinates[i] // xSize)]
                output.append(toAdd)

        return output

    def _create_env(self):
        """Creates an environment and places agent at (0, 0)
        Creates exactly one bomb and cherry somerwhere on the map, as long 
        as they aren't at the origin or on top of each other

        Returns:
            Gridworld -- 5x5 gridworld with a cherry and bomb placed randomly
        """
        params = [self._randomly_create_objects(
            2, (5, 5), reward_map=[1, -1]), 0]
        grid = Gridworld((5, 5), self._ACTION_INFO, params)
        grid.place_agent(0, 0)
        return grid

    def __init__(self):
        """Create the env. Uses an internal variable to store the environment
        """
        self.env = self._create_env()
        self.env.place_agent(0, 0)

        # self.observation_space = spaces.Tuple(
        #     (spaces.Box(low=0, high=4, shape=(5, 5)),
        #      spaces.Box(low=0, high=1, shape=(5, 5))))

        # FLAG - the observation space depends on what we return (see step)

        self.observation_space = spaces.Box(low=0, high=4, shape=(3, 3, 2))

        self.action_space = spaces.Discrete(4)

    def step(self, action):
        """Given an action provided by an agent, return the reward

        Arguments:
            action {int} -- index of the action

        Returns:
            int -- reward to the agent
        """

        done = False

        reward = self.env.move_agent(action)

        # 50 needs to be dependent on _max_episode_steps in __init__.py

        if reward == 1 or self.env._get_epoch() > 50:
            done = True

        # state = (self.env.calculate_grid_map(
        #     (0, 0)), self.env.calculate_contact_map((0, 0)))

        # FLAG - the state affects the observation state

        state = self.env.calculate_distance_matrix((0, 0))

        print(
            f"Action {self._ACTION_BANK[action]} | Epoch : {self.env._get_epoch()}")

        print(
            f"Pos: {self.env._get_agent_coords()} | Reward : {reward} ")

        print()

        return state, reward, done, {}

    def reset(self):
        """Resets environment and re-initializes agent at (0, 0)
        """
        self.env = self._create_env()
        self.env.place_agent(0, 0)
        return self.env.calculate_distance_matrix((0, 0))

    def render(self, mode='human', close=False):
        """Renders an image of the environment currently

        Keyword Arguments:
            mode {str} -- #FLAG ? unknown (default: {'human'})
            close {bool} -- #FLAG ? unknown (default: {False})
            time {int} -- time to display, in milliseconds
        """
        cv2.imshow('image', cv2.resize(self.env.get_representation(showAgent=True), (200, 200),
                                       interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(self._RENDER_TIME)
        cv2.destroyAllWindows()

        return ""
