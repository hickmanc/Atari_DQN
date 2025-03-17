import gymnasium as gym
from convenience import create_env

if __name__ == "__main__":
    env = create_env(
        "ALE/Pong-v5",
        render_mode="rgb_array",
        num_frames = 5
    )
    observations, info = env.reset()
    env = gym.wrappers.HumanRendering(env)
    done = False

    while not done:
        observations, reward, done, truncated, info = env.step(env.action_space.sample())
