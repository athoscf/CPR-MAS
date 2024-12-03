import torch
import torch.nn as nn
import numpy as np
from CommonsGame.model.DeepQNetwork import DeepQNetwork
from CommonsGame.model.ReplayBuffer import ReplayBuffer

class Agent:
    
    def __init__(self, input_dims, num_actions):
        self.Q_network = DeepQNetwork(input_dims, num_actions)
        self.Q_target_network = DeepQNetwork(input_dims, num_actions)

        self.Q_network.init_weights()
        DeepQNetwork.copy_target(self.Q_target_network, self.Q_network)

        self.action_space = [i for i in range(num_actions)]

        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.Q_network.parameters(), lr=5e-4)
        self.gamma = 0.99
        self.decay = 0.99
        self.epsilon = 1
        self.batch_size = 64
        
        self.replay_buffer = ReplayBuffer()

    def e_greedy_policy(self, Qs):
        return np.random.choice(self.action_space) if (np.random.random() <= self.epsilon) else np.argmax(Qs)

    def store_transition(self, state, action, reward, next_state, terminated):
        self.replay_buffer.push(state, action, reward, next_state, terminated, False)

    def learn(self):
        if self.batch_size <= len(self.replay_buffer):
            state_batch, action_batch, reward_batch, next_state_batch, terminated_batch, _ = self.replay_buffer.sample(self.batch_size)
            state_batch = torch.FloatTensor(state_batch).to(self.Q_network.device)
            action_batch = torch.Tensor(action_batch).to(dtype=torch.long).to(self.Q_network.device).unsqueeze(1)
            next_state_batch = torch.FloatTensor(next_state_batch).to(self.Q_network.device)
            reward_batch = torch.FloatTensor(reward_batch).to(self.Q_network.device).unsqueeze(1)
            terminated_batch = torch.FloatTensor(terminated_batch).to(dtype=torch.long).to(self.Q_network.device).unsqueeze(1)
           
            q_targets_next = torch.max(self.Q_target_network(next_state_batch).detach(), dim=1, keepdim=True)[0]
            q_expected = torch.gather(self.Q_network(state_batch), 1, action_batch)
            target = reward_batch + (1 - (terminated_batch)) * self.gamma * q_targets_next

            loss = self.criterion(q_expected, target) 
            self.optimizer.zero_grad()
            loss.backward()
            
            self.optimizer.step()

            DeepQNetwork.soft_update(self.Q_target_network, self.Q_network, 1e-3)
            return loss.detach().cpu().numpy()

    def decay_epsilon(self):
        self.epsilon = max(0.1, self.decay * self.epsilon)

    def choose_action(self, state):
        if np.random.random() <= self.epsilon:
            return np.random.choice(self.action_space)
        
        state = torch.tensor(np.array([state], dtype=np.float32)).to(self.Q_network.device)
        actions = self.Q_network.forward(state)
        return torch.argmax(actions).item()