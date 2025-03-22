# I think the second video is because the agent automatically calls reset
from pathlib import Path
import gymnasium as gym
import numpy as np
import torch
from convenience import create_env, Agent, individual_experience

env_name = "PongNoFrameskip-v4"
device = "mps"
frames_per_experience = 4
mean_reward_to_win = 19.0


if __name__ == "__main__":
    env = create_env(
        env_name,
        render_mode="rgb_array",
        num_frames = frames_per_experience
    )
    env = gym.wrappers.RecordVideo(env, video_folder=Path("."))
    my_agent = Agent(
        env,
        frames_per_experience=frames_per_experience,
        device=device,
        exploration_prob_start=0.0,
        exploration_prob_end=0.0,
        exploration_prob_decay=0.0,
        learning_rate=0.0,
        max_experience_memory=1,
        bellman_discount=0.99
    )
    my_agent.lead_net = torch.load(Path("./winning_model.pt"), weights_only=False)
    my_agent.lead_net.eval()
    # env = gym.wrappers.HumanRendering(env)

    frames_played = 0
    total_rewards = []
    while not my_agent.done:
        frames_played += 1
        reward = my_agent.play_step()
        if reward is not None:
            total_rewards.append(reward)
            mean_reward = np.mean(total_rewards[-100:])
            print(f"{frames_played}: done {len(total_rewards)} games, reward {mean_reward:.3f}, eps {my_agent.current_exploration_prob:.2f}")


    env.close()