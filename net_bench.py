import gymnasium as gym
import numpy as np
from convenience import create_env, experience
from model import my_model
from time import perf_counter
import torch

device = "cpu"
frames_per_experience = int(1e1)
experiences_per_batch = int(1e2)

if __name__ == "__main__":
    model = my_model(num_frames=frames_per_experience, num_actions=6).to(device)
    data = torch.rand(size=(1,frames_per_experience,84,84), device=device, dtype=torch.float)*255
    start = perf_counter()
    print(model(data))
    stop = perf_counter()

    print(f"{(stop - start)}")


