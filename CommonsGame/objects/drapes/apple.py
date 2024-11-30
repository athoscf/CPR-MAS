import numpy as np
from pycolab.prefab_parts import sprites
from pycolab import things as pythings
from scipy.ndimage import convolve
from CommonsGame.constants import *

class Apple(pythings.Drape):
    """Coins Drap"""
    def __init__(self, curtain, character, agentChars, numPadPixels):
        super().__init__(curtain, character)
        self.agentChars = agentChars
        self.numPadPixels = numPadPixels
        self.apples = np.copy(curtain)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        rewards = []
        agentsMap = np.ones(self.curtain.shape, dtype=bool)
        for i in range(len(self.agentChars)):
            rew = self.curtain[things[self.agentChars[i]].position[0], things[self.agentChars[i]].position[1]]
            if rew:
                self.curtain[things[self.agentChars[i]].position[0], things[self.agentChars[i]].position[1]] = False
            rewards.append(rew * 1)
            agentsMap[things[self.agentChars[i]].position[0], things[self.agentChars[i]].position[1]] = False
        the_plot.add_reward(rewards)
        # Matrix of local stock of apples
        kernel = np.array([[1, 1, 1, 1, 1], [1, 1, 1, 1, 1], [1, 1, 0, 1, 1], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]])
        L = convolve(self.curtain[self.numPadPixels + 1:-self.numPadPixels - 1,
                     self.numPadPixels + 1:-self.numPadPixels - 1] * 1, kernel, mode='constant')
        probs = np.zeros(L.shape)
        probs[(L > 0) & (L <= 2)] = respawnProbs[0]
        probs[(L > 2) & (L <= 4)] = respawnProbs[1]
        probs[(L > 4)] = respawnProbs[2]
        appleIdxs = np.argwhere(np.logical_and(np.logical_xor(self.apples, self.curtain), agentsMap))
        for i, j in appleIdxs:
            self.curtain[i, j] = np.random.choice([True, False],
                                                  p=[probs[i - self.numPadPixels - 1, j - self.numPadPixels - 1],
                                                     1 - probs[i - self.numPadPixels - 1, j - self.numPadPixels - 1]])
