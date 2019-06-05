# Test the viability of the gridworld/gym implementation

# from gym import envs
# print(envs.registry.all())

import gym
import kang_gridworld
import numpy as np

# Remember: up/left/down/right corresponds to (0, 1, 2, 3)


def test_1_env_creation():
    # Goal: check that an environment is created and stored
    env = gym.make('kang-grid-v0')

    assert env.env is not None, "Environment not created"
    assert env.env._get_epoch() == 0, "Env not started at 0 epoch"


def test_2_start_at_origin():
    # Goal: check that the agent is initialized at (0, 0)
    env = gym.make('kang-grid-v0')

    assert env.env._get_agent_coords() == (0, 0), "Agent not starting at origin"


def test_3_object_generation():
    # Goal: ensure the bomb and cherry
    # 1. Do not overlap
    # 2. Do not start at the origin
    # 3. Are within the grid

    # Note: this test is stochastic
    for _ in range(1000):
        env = gym.make('kang-grid-v0')

        objects = env.env._get_objects()

        cherry = objects[0]
        bomb = objects[1]

        _, _, cherryX, cherryY = cherry
        _, _, bombX, bombY = bomb

        assert (cherryX, cherryY) != (bombX, bombY), "Cherry and bomb coexist"
        assert (cherryX, cherryY) != (0, 0), "Cherry at origin"
        assert (bombX, bombY) != (0, 0), "Bomb at origin"

        assert cherryX in range(0, 5), "CherryX out of bounds"
        assert cherryY in range(0, 5), "CherryX out of bounds"
        assert bombX in range(0, 5), "CherryX out of bounds"
        assert bombY in range(0, 5), "CherryX out of bounds"


def test_4_movement():
    # Goal: check that the agent can move correctly (in unimpeded situations)
    env = gym.make('kang-grid-v0')

    env.step(3)
    assert env.env._get_agent_coords() == (1, 0), "Agent did not move right"

    env.step(2)
    assert env.env._get_agent_coords() == (1, 1), "Agent did not move down"

    env.step(1)
    assert env.env._get_agent_coords() == (0, 1), "Agent did not move left"

    env.step(0)
    assert env.env._get_agent_coords() == (0, 0), "Agent did not move up"


def test_5_boundaries():
    # Goal: ensure collisions with the edge do not cause the agent to go over
    env = gym.make('kang-grid-v0')

    for _ in range(6):
        env.step(3)
    assert env.env._get_agent_coords() == (
        4, 0), "Agent is not where expected; right"

    for _ in range(6):
        env.step(2)
    assert env.env._get_agent_coords() == (
        4, 4), "Agent is not where expected; down"

    for _ in range(6):
        env.step(1)
    assert env.env._get_agent_coords() == (
        0, 4), "Agent is not where expected; left"

    for _ in range(6):
        env.step(0)
    assert env.env._get_agent_coords() == (
        0, 0), "Agent is not where expected; up"


def test_6_epoch_count():
    # Goal: ensure the epoch counter is correct
    env = gym.make('kang-grid-v0')
    for i in range(50):
        assert env.env._get_epoch() == i, "Count does not match"
        env.step(0)
