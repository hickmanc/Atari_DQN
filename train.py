import gymnasium as gym
import numpy as np
from convenience import create_env, experience
from model import my_model
from time import perf_counter

device = "cpu"
frames_per_experience = 4
experiences_per_batch = int(1e0)

if __name__ == "__main__":
    env = create_env(
        "ALE/Pong-v5",
        render_mode="rgb_array",
        num_frames = frames_per_experience
    )
    model = my_model(num_frames=frames_per_experience, num_actions=env.action_space.n).to(device)
    observations, info = env.reset()
    #env = gym.wrappers.HumanRendering(env)
    my_experience = experience(max_num_experiences=int(1e2), device=device)
    done = False


    frame_count = 0
    start = perf_counter()
    while not done:
        observations, reward, done, truncated, info = env.step(env.action_space.sample())
        observations = np.expand_dims(observations, axis=0)
        my_experience.add_memory(observations)
        random_experiences = my_experience.sample_memory(num_samples=experiences_per_batch)
        frame_count += experiences_per_batch * frames_per_experience

    stop = perf_counter()

    print(f"{frame_count/(stop - start)} FPS")
