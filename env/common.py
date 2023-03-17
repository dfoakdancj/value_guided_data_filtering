from typing import Dict, List, Tuple
import numpy as np
import gym

from value_guided_data_filtering.env.hopper        import HP_Dynamics_Wrapper
from value_guided_data_filtering.env.halfcheetah   import HC_Dynamics_Wrapper
from value_guided_data_filtering.env.ant           import AT_Dynamics_Wrapper
from value_guided_data_filtering.env.walker        import WK_Dynamics_Wrapper

from gym.envs.mujoco.ant_v4         import AntEnv
from gym.envs.mujoco.half_cheetah   import HalfCheetahEnv
from gym.envs.mujoco.walker2d       import Walker2dEnv
from gym.envs.mujoco.hopper         import HopperEnv
from gym.wrappers.time_limit        import TimeLimit

from pathlib import Path


def call_env(env_config: Dict) -> gym.Env:
    if 'Hopper' in env_config['env_name']:
        if env_config['env_name'] == 'Hopper-og':
            return gym.make('Hopper-v2')
        elif env_config['env_name'] == 'Hopper-morph':
            return TimeLimit(
                HopperEnv(xml_file= f"{str(Path(__file__).parent.absolute())}/assets/hopper_morph.xml"),
                1000
            )
        else:
            return HP_Dynamics_Wrapper(param_dict=env_config['param'])
    elif 'HalfCheetah' in env_config['env_name']:
        if env_config['env_name'] == 'HalfCheetah-og':
            return gym.make('HalfCheetah-v2')
        elif env_config['env_name'] == 'HalfCheetah-morph':
            return TimeLimit(
                HalfCheetahEnv(xml_file= f"{str(Path(__file__).parent.absolute())}/assets/halfcheetah_morph.xml",),
                1000
            )
        else:
            return HC_Dynamics_Wrapper(param_dict=env_config['param'])
    elif 'Ant' in env_config['env_name']:
        if env_config['env_name'] == 'Ant-og':
            return gym.make('Ant-v4')
        elif env_config['env_name'] == 'Ant-morph':
            return TimeLimit(
                AntEnv(xml_file= f"{str(Path(__file__).parent.absolute())}/assets/ant_morph.xml",),
                1000
            )
        else:
            return AT_Dynamics_Wrapper(param_dict=env_config['param'])
    elif 'Walker' in env_config['env_name']:
        if env_config['env_name'] == 'Walker-og':
            return gym.make('Walker2d-v2')
        elif env_config['env_name'] == 'Walker-morph':
            return TimeLimit(
                Walker2dEnv(xml_file= f"{str(Path(__file__).parent.absolute())}/assets/walker_morph.xml",),
                1000
            )
        else:
            return WK_Dynamics_Wrapper(param_dict=env_config['param'])
    else:
        raise NotImplementedError


if __name__ == '__main__':
    # env = TimeLimit(
    #         # HumanoidEnv(xml_file= f"{str(Path(__file__).parent.absolute())}/assets/humanoid_test.xml",),
    #         # HalfCheetahEnv(xml_file=f"{str(Path(__file__).parent.absolute())}/assets/halfcheetah_test.xml"),
    #         AntEnv(xml_file= f"{str(Path(__file__).parent.absolute())}/assets/ant_test.xml",),
    #         1000
    #     )

    env = call_env(
        {'env_name': 'Ant-test'}
    )

    for _ in range(100):
        done = False
        s    = env.reset()
        while not done:
            env.render()
            a = np.zeros_like(env.action_space.sample())
            # a = env.action_space.sample()
            a[2] = 0.
            s, r, done, info = env.step(a)