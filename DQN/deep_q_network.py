import torch
import torch.nn as nn
import torch.nn.functional as func

class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
        super(DeepQNetwork, self).__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self.input_dims = input_dims
        self.n_actions = n_actions

        # Two convolutional layers
        self.conv1 = nn.Conv2d(in_channels=input_dims[2], out_channels=32, kernel_size=3, stride=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=2, stride=2)

        # Output layer
        self.output = nn.Linear(32, n_actions)  # Output layer with 8 units, one for each action

        self.to(self.device)

    def forward(self, s):
        s = s.permute(0, 3, 1, 2)  # Change the dimension order to match Conv2D expectations
        features = func.relu(self.conv1(s))
        features = func.relu(self.conv2(features))
        features = torch.flatten(features, 1)  # Flatten the feature maps
        return self.output(features)

    def init_weights(self):
        def init_layer_weights(m):
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight)

        self.apply(init_layer_weights)

    @staticmethod
    def copy_target(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    @staticmethod
    def soft_update(target_model, local_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
