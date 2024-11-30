import numpy as np
from pycolab import things as pythings
from scipy.ndimage import convolve
from CommonsGame.constants import *

class Apple(pythings.Drape):
    
    def __init__(self, curtain, character, agent_chars, num_pad_pixels):
        super().__init__(curtain, character)
        self.agent_chars = agent_chars
        self.num_pad_pixels = num_pad_pixels
        self.apples = np.copy(curtain)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        rewards = self.calculate_rewards(things)
        the_plot.add_reward(rewards)

        available_cells = self.available_cells(things)
        self.respawn_apples(available_cells, layers)

    def agents(self, things):
        return [things[agent] for agent in self.agent_chars]
        
    def collected_apple(self, position):
        return self.curtain[position[0], position[1]]

    def remove_apple(self, position):
        self.curtain[position[0], position[1]] = False

    def available_cells(self, things):
        available_cells = np.ones(self.curtain.shape, dtype=bool)

        for agent in self.agents(things):
            available_cells[agent.position[0], agent.position[1]] = False

        return available_cells

    def calculate_rewards(self, things):
        rewards = []
        for agent in self.agents(things):
            if self.collected_apple(agent.position):
                rewards.append(1)
                self.remove_apple(agent.position)
            else:
                rewards.append(0)
        return rewards
        
    def respawn_apples(self, available_cells, layers):
        kernel = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        nr_apples = convolve(self.curtain[self.num_pad_pixels + 1:-self.num_pad_pixels - 1,
                     self.num_pad_pixels + 1:-self.num_pad_pixels - 1] * 1, kernel, mode='constant')
        
        probs = np.zeros(nr_apples.shape)
        probs[(nr_apples > 0) & (nr_apples <= 2)] = RespawnProbabilities.LOW
        probs[(nr_apples > 2) & (nr_apples <= 4)] = RespawnProbabilities.MEDIUM
        probs[(nr_apples > 4)] = RespawnProbabilities.HIGH
        
        apple_idxs = np.argwhere(np.logical_and(
            np.logical_and(np.logical_xor(self.apples, self.curtain), available_cells), 
            np.logical_not(layers[Sprites.WALL]) 
        ))
        
        for i, j in apple_idxs:
            x = i - self.num_pad_pixels - 1
            y = j - self.num_pad_pixels - 1
            
            self.curtain[i, j] = np.random.choice(
                [True, False],
                p=[probs[x, y], 1 - probs[x, y]])
        
