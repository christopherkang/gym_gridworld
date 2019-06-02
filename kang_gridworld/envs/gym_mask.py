from .gridworld import Gridworld
import numpy as np

import gym
from gym import error, spaces, utils
from gym.utils import seeding


class Kang_Grid(gym.Env):
    metadata = {'render.modes': ['human']}
    _ACTION_BANK = ["UP", "LEFT", "DOWN", "RIGHT"]
    _ACTION_DEF = [[0, -1],
                [-1, 0],
                [0, 1],
                [1, 0]]
    _ACTION_INFO = [_ACTION_BANK, _ACTION_DEF]

    def _randomly_create_objects(number_of_objects, xyDimension,
                            reward=None, reward_map=None):
        """Creates random objects matching 'parameters' format
        
        Arguments:
            number_of_objects {int} -- number of objects to be generated
            reward {int} -- reward of individual object
            xyDimension {tuple} -- dimension of world
        
        Keyword Arguments:
            reward_map {array of ints} -- list of rewards to use (default: {""})
        
        Returns:
            list -- list of objects
        """
        xSize = xyDimension[0]
        ySize = xyDimension[1]

        coordinates = np.random.choice(xSize * ySize - 1, number_of_objects, replace=False)

        output = []

        if reward_map:
            for i in range(number_of_objects):
                toAdd = [reward_map[i], True, coordinates & xSize, coordinates // xSize]
                output.append(toAdd)
        else:
            for _ in range(number_of_objects):
                toAdd = [reward, True, coordinates & xSize, coordinates // xSize]
                output.append(toAdd)

        return output

    def _create_env(self):
        params = [self._randomly_create_objects(2, (5, 5), reward_map=[1, -1]), 0]
        return Gridworld((5,5), _ACTION_INFO, params)


    def __init__(self):
        """Create the env. Uses an internal variable 
        """
        self.env = self._create_env()
        self.env.place_agent(0, 0)

    def step(self, action):
        self.env.move_agent(action)

    def reset(self):
        self.env = self._create_env()

    def render(self, mode='human', close=False):
        return super().render(mode=mode)