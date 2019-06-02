from gym.envs.registration import register

register(
    id='kang-grid-v0',
    entry_point='gym_foo.envs:GridEnv',
)