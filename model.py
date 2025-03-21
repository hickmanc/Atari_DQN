import torch
import torch.nn as nn

class DQN(nn.Module):
    def __init__(self, num_frames: int, num_actions: int, *args, **kwargs):
        super(DQN, self).__init__()
        self.convolutions = nn.Sequential(
            nn.Conv2d(num_frames, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        dummy_input = torch.zeros(size=(1, num_frames, 84, 84))
        convolutions_output_shape = self.convolutions(dummy_input).size()[-1]

        self.fully_connected_nn = nn.Sequential(
            nn.Linear(convolutions_output_shape, 512),
            nn.ReLU(),
            nn.Linear(512, num_actions)
        )

    def forward(self, observation_tensors: torch.ByteTensor):
        observation_tensors = torch.tensor(observation_tensors, dtype=torch.float)
        observation_tensors /= 255.0
        result = self.convolutions(observation_tensors)
        result = self.fully_connected_nn(result)
        return result