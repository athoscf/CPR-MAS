
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity=int(1e6), seed=42):
        random.seed(seed)
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, terminated, truncated):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, terminated, truncated)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        batch = random.sample(self.buffer, min(len(self.buffer), batch_size))
        state, action, reward, next_state, terminated, truncated = map(np.stack, zip(*batch))
        return state, action, reward, next_state, terminated, truncated

    def __len__(self):
        return len(self.buffer)