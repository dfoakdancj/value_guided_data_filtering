from typing import List, Dict, Tuple
import numpy as np
import torch
import os
import gym
import yaml
import torch.nn as nn
import datetime


def current_time() -> str:
    now = datetime.now()
    return now.strftime("%m-%d-%H-%M")


def soft_update(src_model: nn.Module, tar_model: nn.Module, tau: float) -> None:
    for param_src, param_tar in zip(src_model.parameters(), tar_model.parameters()):
        param_tar.data.copy_(tau * param_src.data + (1 - tau) * param_tar.data)


def confirm_path_exist(path: str) -> None:
    if not os.path.exists(path):
        os.makedirs(path)


def seed_all(seed: int, env: gym.Env = None) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if env:
        env.seed(seed)


def make_exp_path(config: Dict, exp_name: str) -> Dict:
    exp_path = config['result_path'] + f"{config['env']}"
    if exp_name is not None:
        exp_path += f'-{exp_name}'
    exp_path += f"-{config['seed']}"
    while os.path.exists(exp_path + f"/"):
        exp_path += '_*'
    exp_path += "/"
    confirm_path_exist(exp_path)
    confirm_path_exist(f'{exp_path}model/')
    config.update({'exp_path': exp_path})
    with open(config['exp_path'] + 'config.yaml', 'w', encoding='utf-8') as f:
        yaml.safe_dump(config, f, indent=2)
    return config