import gymnasium as gym
import ale_py
from wrappers import MultiFrame, Atari_img_to_torch
from stable_baselines3.common import atari_wrappers
import numpy as np
from random import sample
import torch
from model import DQN

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
    

class enhanced_experience:
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
    
class Agent:
    def __init__(
            self,
            env: gym.Env,
            frames_per_experience: int,
            device: str,
            exploration_prob_start: float,
            exploration_prob_end: float,
            exploration_prob_decay: float,
            learning_rate: float
        ):
        self.env = env
        self.experiences = []
        self.state = None
        self.exploration_prob_start = exploration_prob_start
        self.exploration_prob_end = exploration_prob_end
        self.exploration_prob_decay = exploration_prob_decay
        self.current_exploration_prob = exploration_prob_start
        self.learning_rate = learning_rate
        self.lead_net = DQN(
            num_actions=self.env.action_space.n,
            num_frames=frames_per_experience
        ).to(device)
        self.optimizer = torch.optim.Adam(
            self.lead_net.parameters(), lr=self.learning_rate
        )
        self.stable_net = DQN(
            num_actions=self.env.action_space.n,
            num_frames=frames_per_experience
        ).to(device)

    def reset(self):
        self.state, info = self.env.reset()
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self):
        done_reward = None

        if np.random.random() < self.current_exploration_prob:
            action = self.env.action_space.sample()
        else:
            predicted_q_vals = self.lead_net(self.state)
            _, action = torch.max(predicted_q_vals, dim=1)
            action = int(action.item())

        self.current_exploration_prob = max(
            self.current_exploration_prob * self.exploration_prob_decay,
            self.exploration_prob_end
        )