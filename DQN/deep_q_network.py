import torch
import torch.nn as nn
import torch.nn.functional as func

class DeepQNetwork(nn.Module):
    def __init__(self, input_dims, n_actions):
      super(DeepQNetwork, self).__init__()

      self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
      self.input_dims = input_dims
      self.n_actions = n_actions

      self.conv1 = nn.Conv2d(in_channels=input_dims[2], out_channels=32, kernel_size=3, stride=3)
      self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=2)
      self.conv3 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=1, stride=1)
      self.fc1 = nn.Linear(64, 64)
      self.Q_network = nn.Linear(64, n_actions)

      self.to(self.device)

    def forward(self, s):
        s = s.permute(0, 3, 1, 2)
        features = func.relu(self.conv1(s))
        features = func.relu(self.conv2(features))
        features = func.relu(self.conv3(features))
        features = torch.flatten(features, 1)
        features = func.relu(self.fc1(features))
        return self.Q_network(features)

    def init_weights(self):
        def init_layer_weights(m):
            if type(m) == nn.Linear:
                nn.init.orthogonal_(m.weight)
                
        self.apply(init_layer_weights)

    def copy_target(target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def soft_update(target_model, local_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)