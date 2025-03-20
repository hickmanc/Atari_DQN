import gymnasium as gym
import ale_py
from wrappers import MultiFrame, Atari_img_to_torch
from stable_baselines3.common import atari_wrappers
import numpy as np
from random import sample
import torch
from model import DQN
from dataclasses import dataclass

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

@dataclass
class individual_experience:
    starting_state: torch.Tensor
    resulting_state: torch.Tensor
    reward: float
    action_take: int
    done: bool

'''
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
'''
    

class experiences_holder:
    def __init__(self, max_num_experiences):
        self.max_num_experiences = max_num_experiences
        self.memories = []

    def add_memory(self, new_memory):
        if len(self.memories) >= self.max_num_experiences:
            del self.memories[0]
        self.memories.append(new_memory)

    def sample_memory(self, num_samples):
        num_samples = min(num_samples, len(self.memories))
        return sample(self.memories, num_samples)
    
    def get_num_experiences(self):
        return len(self.memories)
    
class Agent:
    def __init__(
            self,
        env: gym.Env,
        max_experience_memory: int,
        frames_per_experience: int,
        device: str,
        exploration_prob_start: float,
        exploration_prob_end: float,
        exploration_prob_decay: float,
        learning_rate: float,
        bellman_discount: float
    ):
        self.env = env
        self.device = device
        self.state = None
        self.exploration_prob_start = exploration_prob_start
        self.exploration_prob_end = exploration_prob_end
        self.exploration_prob_decay = exploration_prob_decay
        self.current_exploration_prob = exploration_prob_start
        self.learning_rate = learning_rate
        self.bellman_discout = bellman_discount
        self.num_plays = 0
        self.loss_fn = torch.nn.MSELoss()
        self.experiences = experiences_holder(
            max_num_experiences=max_experience_memory
        )
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
        self.reset()



        # fill replay buffer with experiences
        while self.experiences.get_num_experiences() < max_experience_memory:
            self.play_step()


    def reset(self):
        self.state, info = self.env.reset()
        self.state = np.expand_dims(self.state, axis=0)
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self):
        self.num_plays += 1
        done_reward = None

        # select the action
        if np.random.random() < self.current_exploration_prob:
            action = self.env.action_space.sample()
        else:
            state_tensor = torch.from_numpy(self.state).to(dtype=torch.float, device=self.device)
            predicted_q_vals = self.lead_net(state_tensor)
            _, action = torch.max(predicted_q_vals, dim=1)
            action = int(action.item())

        old_state = self.state.copy()
        # take the action
        self.state, reward, done, truncated, _ = self.env.step(action=action)
        # update the accumulated reward
        self.total_reward += reward
        # make sure the state is the correct shape for batches
        self.state = np.expand_dims(self.state, axis=0)
        self.experiences.add_memory(
            individual_experience(
                starting_state=old_state,
                resulting_state=self.state.copy(),
                reward=float(reward),
                action_take=action,
                done=done
            )
        )

        # update exploration probability
        self.current_exploration_prob = max(
            self.exploration_prob_start - self.num_plays/self.exploration_prob_decay,
            self.exploration_prob_end
        )

        if done or truncated:
            done_reward = self.total_reward
            self.reset()
        return done_reward
    
    # batch is a list of individual_experiences
    def calc_loss(self, batch: list[individual_experience]):
        starting_states = np.concat([e.starting_state for e in batch], axis=0)
        starting_states = torch.from_numpy(starting_states).to(dtype=torch.float, device=self.device)
        resulting_states = np.concat([e.resulting_state for e in batch], axis=0)
        resulting_states = torch.from_numpy(resulting_states).to(dtype=torch.float, device=self.device)
        # list of booleans specifying whether the game was done after this experience
        experience_done = [e.done for e in batch]
        chosen_actions = [e.action_take for e in batch]
        rewards = torch.tensor(
            [e.reward for e in batch],
            device=self.device,
            dtype=torch.float
        )
        predicted_q_values = self.lead_net(starting_states)
        chosen_q_values = predicted_q_values[torch.arange(len(chosen_actions)),chosen_actions]
        # only an estimation of the actual q values, but apparantly it works
        with torch.no_grad():
            next_state_predicted_values = self.stable_net(resulting_states).max(1)[0]
            # this is guaranteed to be true beacuse the reward after done is 0
            next_state_predicted_values[experience_done] = 0.0
            next_state_predicted_values = next_state_predicted_values.detach()

        expected_q_values = rewards + next_state_predicted_values * self.bellman_discout
        return self.loss_fn(chosen_q_values, expected_q_values)
        