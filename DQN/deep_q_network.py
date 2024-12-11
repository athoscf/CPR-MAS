import torch
import torch.nn as nn
import torch.nn.functional as func

class DeepQNetwork(nn.Module):
    def __init__(self, input_shape, n_actions):
        super(DeepQNetwork, self).__init__()

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.input_shape = input_shape # Shape of the input, e.g., (3, 64, 64)
        self.n_actions = n_actions

        # Dynamically calculate the input size for the first linear layer
        sample_input = torch.zeros(self.input_shape)  # Create a dummy tensor with input shape
        input_size = self.calculate_input_size(sample_input)

        # Define fully connected layers
        self.fc1 = nn.Linear(input_size, 32)  # First hidden layer with 32 units
        self.fc2 = nn.Linear(32, 32)         # Second hidden layer with 32 units
        self.output = nn.Linear(32, n_actions)  # Output layer with 8 units (n_actions)

        self.to(self.device)

    def forward(self, s):
        # Flatten the input tensor from (batch_size, channels, height, width) to (batch_size, -1)
        x = torch.flatten(s, start_dim=1)  
        x = func.relu(self.fc1(x))  # First hidden layer with ReLU activation
        x = func.relu(self.fc2(x)) # Second hidden layer with ReLU activation
        return self.output(x)      # Output layer

    @staticmethod
    def calculate_input_size(sample_input):
        """
        Calculate the flattened size of the input for linear layers.
        
        Args:
            sample_input (torch.Tensor): A sample input tensor with shape 
                                         (channels, height, width).
        
        Returns:
            int: Flattened size of the input tensor.
        """
        return torch.flatten(sample_input, start_dim=0).shape[0]

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
