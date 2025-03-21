import gymnasium as gym
import numpy as np
from collections import deque

# pytorch expects the first channel to be color
# the atari environments return color as the third channel
# this changes that
class Atari_img_to_torch(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        new_observation_shape = [
            self.observation_space.shape[-1],
            self.observation_space.shape[0],
            self.observation_space.shape[1]
        ]

        self.observation_space = gym.spaces.Box(
            low=self.observation_space.low.min(),
            high=self.observation_space.high.max(),
            shape=new_observation_shape,
            dtype=self.observation_space.dtype
        )

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)
    

# this will store multiple frames in a single observation
class MultiFrame(gym.ObservationWrapper):
    def __init__(self, env, num_frames):
        super().__init__(env)
        self.frames = deque(maxlen=num_frames)
        # update the observation space of the environment
        # to match our multi frame observations
        self.observation_space = gym.spaces.Box(
            env.observation_space.low.repeat(num_frames, axis=0),
            env.observation_space.high.repeat(num_frames, axis=0),
            dtype=self.observation_space.dtype
        )

    # fill the frames with dummy values to start
    # for example, if I want a 5 frame history, that doesn't exist on frame 1
    def reset(self, *, seed = None, options = None):
        for _ in range(self.frames.maxlen):
            self.frames.append(self.env.observation_space.low)
        observation, info = self.env.reset()
        return self.observation(observation), info
    
    def observation(self, observation):
        self.frames.append(observation)
        return np.concatenate(self.frames)
    