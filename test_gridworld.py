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
    for _ in range(10):
        env = gym.make('kang-grid-v0')

        print(type(env.env))

        objects = env.env.env._get_objects()

        cherry = objects[0]
        bomb = objects[1]

        _, _, cherryX, cherryY = cherry
        _, _, bombX, bombY = bomb

        assert isinstance(cherryX, np.int64), "CherryX is not an int"
        assert isinstance(cherryY, np.int64), "CherryX is not an int"
        assert isinstance(bombX, np.int64), "CherryX is not an int"
        assert isinstance(bombY, np.int64), "CherryX is not an int"

        assert cherryX != bombX or cherryY != bombY, "Cherry and bomb coexist"
        assert cherryX != 0 or cherryY != 0, "Cherry at origin"
        assert bombX != 0 or bombY != 0, "Bomb at origin"

        assert cherryX in range(0, 5), "CherryX out of bounds"
        assert cherryY in range(0, 5), "CherryX out of bounds"
        assert bombX in range(0, 5), "CherryX out of bounds"
        assert bombY in range(0, 5), "CherryX out of bounds"


def test_4_movement():
    # Goal: check that the agent can move correctly (in unimpeded situations)
    env = gym.make('kang-grid-v0')

    env.step(3)
    assert env.env.env._get_agent_coords() == (1, 0), "Agent did not move right"

    env.step(2)
    assert env.env.env._get_agent_coords() == (1, 1), "Agent did not move down"

    env.step(1)
    assert env.env.env._get_agent_coords() == (0, 1), "Agent did not move left"

    env.step(0)
    assert env.env.env._get_agent_coords() == (0, 0), "Agent did not move up"


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


def test_7_env_output():
    env = gym.make('kang-grid-v0')
    state, _, _, _ = env.step(0)
    assert state.shape == (5, 5, 2), "Output does not match expected shape"


def test_8_contact_map():
    for _ in range(1000):
        env = gym.make('kang-grid-v0')

        env.env._set_render_time(200)

        objects = env.env.env._get_objects()

        agentX, agentY = env.env.env._get_agent_coords()

        contacts = env.env.env.calculate_grid_map()[:, :, 2]

        cherry = objects[0]
        bomb = objects[1]

        _, _, cherryX, cherryY = cherry
        _, _, bombX, bombY = bomb

        def is_contacting(x1, y1, x2, y2):
            if (abs(x1-x2) + abs(y1-y2) <= 1):
                return True
            else:
                return False

        if (is_contacting(agentX, agentY, cherryX, cherryY)):
            assert contacts[1][0] == 1, "The contacting cherry is not marked"
        else:
            assert contacts[1][0] == 0, "The noncontacting cherry is marked"

        if (is_contacting(agentX, agentY, bombX, bombY)):
            assert contacts[2][0] == 1, "The contacting bomb is not marked"
        else:
            assert contacts[2][0] == 0, "The noncontacting bomb is marked"

        if (is_contacting(cherryX, cherryY, bombX, bombY)):
            assert contacts[2][1] == 1, "The bomb/cherry pair is not marked"
        else:
            assert contacts[2][1] == 0, "The bomb/cherry pair is marked"
