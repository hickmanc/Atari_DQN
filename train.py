import gymnasium as gym
import numpy as np
import torch
from convenience import create_env, Agent, individual_experience

env_name = "ALE/Pong-v5"
device = "mps"
frames_per_experience = 4
experiences_per_batch = int(1e0)
gamma = 0.99
batch_size = 32
learning_rate = 1e-4
sync_interval = int(1e3)
exploration_prob_decay = int(15e4)
exploration_prob_start = 1.0
exploration_prob_end = 0.01
max_experience_memory = int(1e2)


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
        learning_rate=learning_rate,
        max_experience_memory=max_experience_memory,
        bellman_discount=gamma
    )
    env = gym.wrappers.HumanRendering(env)
    # my_experience = experience(max_num_experiences=int(1e2), device=device)
    done = False

    frames_played = 0
    while not done:
        my_agent.play_step()
        random_experiences = my_agent.experiences.sample_memory(2)
        # why does commenting out this line lead to such a speed boost????
        #print(random_experiences)
        my_agent.lead_net.zero_grad()
        # performs forward pass for prediction and calculates loss
        loss = my_agent.calc_loss(random_experiences)
        loss.backward()
        my_agent.optimizer.step()
        if (frames_played % sync_interval) == 0:
            my_agent.stable_net.load_state_dict(my_agent.lead_net.state_dict())
        frames_played += 1
        print(frames_played)
        # observations = np.expand_dims(observations, axis=0)
        # q_value_predictions = lead_net(observations)
        #my_experience.add_memory(observations)
        #random_experiences = my_experience.sample_memory(num_samples=experiences_per_batch)
        #print(random_experiences.shape)

