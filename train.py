import gymnasium as gym
import numpy as np
from convenience import create_env, experience
from model import my_model

if __name__ == "__main__":
    env = create_env(
        "ALE/Pong-v5",
        render_mode="rgb_array",
        num_frames = 5
    )
    model = my_model(num_frames=5, num_actions=env.action_space.n, device="cpu")
    observations, info = env.reset()
    env = gym.wrappers.HumanRendering(env)
    my_experience = experience(max_num_experiences=int(1e2))
    done = False


    while not done:
        observations, reward, done, truncated, info = env.step(env.action_space.sample())
        observations = np.expand_dims(observations, axis=0)
        my_experience.add_memory(observations)
        random_experiences = my_experience.sample_memory(num_samples=10)
        print(model(random_experiences))
