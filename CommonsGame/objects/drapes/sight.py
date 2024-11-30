import numpy as np
from pycolab.prefab_parts import sprites
from pycolab import things as pythings
from CommonsGame.constants import *

class Sight(pythings.Drape):

    def __init__(self, curtain, character, agent_chars, num_pad_pixels):
        super().__init__(curtain, character)
        self.agent_chars = agent_chars
        self.h = curtain.shape[0] - (num_pad_pixels * 2 + 2)
        self.w = curtain.shape[1] - (num_pad_pixels * 2 + 2)

    def update(self, actions, board, layers, backdrop, things, the_plot):
        np.logical_and(self.curtain, False, self.curtain)
        ags = [things[c] for c in self.agent_chars]
        for agent in ags:
            if agent.visible:
                pos = agent.position
                if agent.orientation == Orientations.NORTH:
                    self.curtain[pos[0] - 1, pos[1]] = True
                elif agent.orientation == Orientations.EAST:
                    self.curtain[pos[0], pos[1] + 1] = True
                elif agent.orientation == Orientations.SOUTH:
                    self.curtain[pos[0] + 1, pos[1]] = True
                elif agent.orientation == Orientations.WEST:
                    self.curtain[pos[0], pos[1] - 1] = True
                self.curtain[:, :] = np.logical_and(self.curtain, np.logical_not(layers['=']))
