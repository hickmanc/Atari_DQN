import gymnasium as gym
import ale_py
from wrappers import MultiFrame, Atari_img_to_torch
from stable_baselines3.common import atari_wrappers

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