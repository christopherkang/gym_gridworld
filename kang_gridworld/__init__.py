from gym.envs.registration import register

register(
    id='kang-grid-v0',
    entry_point='kang_gridworld.envs:KangGrid',
)
