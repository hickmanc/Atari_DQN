import gymnasium as gym
import numpy as np
import torch
from convenience import create_env, experience, Agent
from time import perf_counter
from dataclasses import dataclass

env_name = "ALE/Pong-v5"
device = "cpu"
frames_per_experience = 4
experiences_per_batch = int(1e0)
gamma = 0.99
batch_size = 32
learning_rate = 1e-4
sync_interval = int(1e3)
replay_start_size = int(1e4)
exploration_prob_decay = int(15e4)
exploration_prob_start = 1.0
exploration_prob_end = 0.01

@dataclass
class individual_experience:
    starting_state: torch.Tensor
    resulting_state: torch.Tensor
    reward: float
    action_take: int
    done: bool


if __name__ == "__main__":
    env = create_env(
        env_name,
        render_mode="rgb_array",
        num_frames = frames_per_experience
    )
    my_agent = Agent(
        env,
        frames_per_experience=frames_per_experience,
        device=device,
        exploration_prob_start=exploration_prob_start,
        exploration_prob_end=exploration_prob_end,
        exploration_prob_decay=exploration_prob_decay,
        learning_rate=learning_rate
    )
    observations, info = env.reset()
    #env = gym.wrappers.HumanRendering(env)
    # my_experience = experience(max_num_experiences=int(1e2), device=device)
    experiences = experience(max_num_experiences=10, device=device)
    done = False


    while not done:
        past_state = observations.copy()

        observations, reward, done, truncated, info = env.step(env.action_space.sample())
        observations = np.expand_dims(observations, axis=0)
        # q_value_predictions = lead_net(observations)
        new_experience = individual_experience(
            starting_state=torch.tensor(observations, device=device, dtype=torch.float),
            resulting_state=torch.tensor([]),
            reward=reward,
            action_take=0,
            done=done
        )
        #my_experience.add_memory(observations)
        #random_experiences = my_experience.sample_memory(num_samples=experiences_per_batch)
        #print(random_experiences.shape)

