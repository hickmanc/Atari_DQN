from pathlib import Path
import numpy as np
import torch
from convenience import create_env, Agent

env_name = "BreakoutNoFrameskip-v4"
device = "mps"
frames_per_experience = 5
gamma = 0.99
batch_size = 32
max_experience_memory = int(2e4)
learning_rate = 1e-4
sync_interval = int(1e3)
exploration_prob_decay = int(15e4)
exploration_prob_start = 1.0
exploration_prob_end = 0.01
mean_reward_to_win = 19.0


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
    # env = gym.wrappers.HumanRendering(env)

    frames_played = 0
    total_rewards = []
    while True:
        frames_played += 1
        my_agent.current_exploration_prob = max(
            my_agent.exploration_prob_end,
            my_agent.exploration_prob_start - frames_played / my_agent.exploration_prob_decay
        )
        #print(my_agent.current_exploration_prob)
        reward = my_agent.play_step()
        if reward is not None:
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-100:])
            if len(total_rewards) % 15 == 0:
                print(f"{frames_played}: done {len(total_rewards)} games, reward {mean_reward:.3f}, eps {my_agent.current_exploration_prob:.2f}")
            if mean_reward > mean_reward_to_win:
                torch.save(my_agent.lead_net, Path("./winning_model.pt"))
                break

        if my_agent.experiences.get_num_experiences() < max_experience_memory:
            continue
        if (frames_played % sync_interval) == 0:
            my_agent.stable_net.load_state_dict(my_agent.lead_net.state_dict())

        my_agent.lead_net.zero_grad()
        random_experiences = my_agent.experiences.sample_memory(batch_size)
        # performs forward pass for prediction and calculates loss
        loss = my_agent.calc_loss(random_experiences)
        loss.backward()
        my_agent.optimizer.step()
    env.close()