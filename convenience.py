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
        #indices = np.random.choice(len(self.memories), num_samples, replace=False)
        #return [self.memories[i] for i in indices]

    
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
        self.experiences = []
        self.state = None
        self.exploration_prob_start = exploration_prob_start
        self.exploration_prob_end = exploration_prob_end
        self.exploration_prob_decay = exploration_prob_decay
        self.current_exploration_prob = exploration_prob_start
        self.learning_rate = learning_rate
        self.bellman_discout = bellman_discount
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
        self._reset()

    def _reset(self):
        self.state, _ = self.env.reset()
        self.state = np.expand_dims(self.state, axis=0)
        self.state = torch.tensor(
            self.state,
            device=self.device,
            dtype=torch.float
        )
        self.total_reward = 0.0

    @torch.no_grad()
    def play_step(self):
        done_reward = None
        # select the action
        if np.random.random() < self.current_exploration_prob:
            action = self.env.action_space.sample()
        else:
            #print(self.state.shape)
            predicted_q_vals = self.lead_net(self.state)
            _, action = torch.max(predicted_q_vals, dim=1)
            action = int(action.item())

        # take the action
        new_state, reward, done, truncated, _ = self.env.step(action=action)
        # update the accumulated reward
        self.total_reward += reward
        # make sure the state is the correct shape for batches
        new_state = np.expand_dims(new_state, axis=0)
        # make sure the state is saved as a tensor
        new_state = torch.tensor(
            new_state,
            device=self.device,
            dtype=torch.float
        )
        exp = individual_experience(
                starting_state=self.state.clone().detach(),
                resulting_state=new_state.clone().detach(),
                reward=float(reward),
                action_take=action,
                done=(done or truncated)
            )
        self.experiences.add_memory(exp)
        self.state = new_state.clone().detach()
        if done or truncated:
            print(f"Done: {done}, Trunc: {truncated}")
            done_reward = self.total_reward
            self._reset()
        return done_reward
    
    # batch is a list of individual_experiences
    def calc_loss(self, batch: list[individual_experience]):
        starting_states, resulting_states, dones, actions, rewards = [], [], [], [], []
        for e in batch:
            starting_states.append(e.starting_state)
            resulting_states.append(e.resulting_state)
            dones.append(e.done)
            actions.append(e.action_take)
            rewards.append(e.reward)
        starting_states = torch.concat(starting_states, dim=0)
        resulting_states = torch.concat(resulting_states, dim=0)
        actions = torch.tensor(actions, device=self.device)
        # list of booleans specifying whether the game was done after this experience
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float)
        state_action_values = self.lead_net(starting_states).gather(
            1, actions.unsqueeze(-1)
        ).squeeze(-1)
        #chosen_q_values = predicted_q_values[torch.arange(len(actions)), actions]
        # only an estimation of the actual q values, but apparantly it works
        with torch.no_grad():
            next_state_values = self.stable_net(resulting_states).max(1)[0]
            # this is guaranteed to be true beacuse the reward after done is 0
            next_state_values[dones] = 0.0
            next_state_values = next_state_values.detach()

        expected_q_values = rewards + next_state_values * self.bellman_discout
        return self.loss_fn(state_action_values, expected_q_values)
        