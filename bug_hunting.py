import kang_gridworld 
import numpy as np
import gym


while True:
    env = gym.make('kang-grid-v0')

    print(env.env.env._get_objects())

    print()