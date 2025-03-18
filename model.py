import torch
import torch.nn as nn

class my_model(nn.Module):
    def __init__(self, num_frames: int, num_actions: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.convolutions = nn.Sequential(
            nn.Conv2d(in_channels=num_frames, out_channels=32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten()
        )

        dummy_input = torch.zeros(size=(1, num_frames, 84, 84))
        convolutions_output_shape = self.convolutions(dummy_input).shape[-1]

        self.fully_connected_nn = nn.Sequential(
            nn.Linear(in_features=convolutions_output_shape, out_features=512),
            nn.ReLU(),
            nn.Linear(in_features=512, out_features=num_actions)
        )

    def forward(self, observation_tensors):
        observation_tensors = torch.tensor(observation_tensors, dtype=torch.float)
        observation_tensors /= 255.0
        result = self.convolutions(observation_tensors)
        result = self.fully_connected_nn(result)
        return result