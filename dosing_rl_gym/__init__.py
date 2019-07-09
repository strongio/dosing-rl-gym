from gym.envs.registration import register

register(
    id='Diabetic-v0',
    entry_point='dosing_rl_gym.envs:Diabetic0Env'
)

register(
    id='Diabetic-v1',
    entry_point='dosing_rl_gym.envs:Diabetic1Env'
)

