import gymnasium as gym
import ale_py
from wrappers import MultiFrame, Atari_img_to_torch
from stable_baselines3.common import atari_wrappers
import numpy as np
from random import sample
import torch

def create_env(env_name: str, num_frames: int, **kwargs):
    env = gym.make(env_name, **kwargs)
    env = atari_wrappers.AtariWrapper(
        env=env,
        clip_reward=False,
        noop_max=0
    )
    env = Atari_img_to_torch(env)
    env = MultiFrame(env, num_frames=num_frames)
    return env


class experience:
    def __init__(self, max_num_experiences, device: str):
        self.max_num_experiences = max_num_experiences
        self.memories = []
        self.device = device

    def add_memory(self, new_memory):
        if len(self.memories) >= self.max_num_experiences:
            del self.memories[0]
        new_memory = torch.tensor(new_memory, device=self.device, dtype=torch.float)
        self.memories.append(new_memory)

    def sample_memory(self, num_samples):
        num_samples = min(num_samples, len(self.memories))
        return torch.concat(sample(self.memories, num_samples), axis=0)