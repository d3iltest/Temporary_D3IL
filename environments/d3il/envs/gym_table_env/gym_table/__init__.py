from gym.envs.registration import register

register(
    id="table-v0",
    entry_point="gym_table.envs:Table_Env",
    max_episode_steps=500,
)
