import gymnasium as gym
import numpy as np
import torch
from convenience import create_env, Agent, individual_experience

env_name = "PongNoFrameskip-v4"
device = "mps"
frames_per_experience = 4
gamma = 0.99
batch_size = 32
learning_rate = 1e-4
sync_interval = int(1e3)
max_experience_memory = int(1e4)
# max_experience_memory = 5
# episolon is exploration probability
epsilon_decay_last_frame = int(15e4)
epsilon_start = 1.0
epsilon_final = 0.01


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
        exploration_prob_start=epsilon_start,
        exploration_prob_end=epsilon_final,
        exploration_prob_decay=epsilon_decay_last_frame,
        learning_rate=learning_rate,
        max_experience_memory=max_experience_memory,
        bellman_discount=gamma
    )
    #env = gym.wrappers.HumanRendering(env)

    total_rewards = []
    while True:
        reward = my_agent.play_step()
        if reward is not None:
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-100:])
            print(mean_reward)
            print(my_agent.current_exploration_prob)
        random_experiences = my_agent.experiences.sample_memory(batch_size)
        # why does commenting out this line lead to such a speed boost????
        #print(random_experiences)
        my_agent.lead_net.zero_grad()
        # performs forward pass for prediction and calculates loss
        loss = my_agent.calc_loss(random_experiences)
        loss.backward()
        my_agent.optimizer.step()
        if (my_agent.num_plays % sync_interval) == 0:
            my_agent.stable_net.load_state_dict(my_agent.lead_net.state_dict())

