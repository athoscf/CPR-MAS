import numpy as np
from pycolab import things as pythings
from CommonsGame.resources.constants import *

class Scope(pythings.Drape):

    def __init__(self, curtain, character, agent_chars):
        super().__init__(curtain, character)
        self.agent_chars = agent_chars
        self.scope_offset = {
            Orientations.NORTH: (-1, 0),
            Orientations.EAST: (0, 1),
            Orientations.SOUTH: (1, 0),
            Orientations.WEST: (0, -1)
        }

    def update(self, actions, board, layers, backdrop, things, the_plot):
        if actions is None: return
        
        np.logical_and(self.curtain, False, self.curtain)
        agents = [things[c] for c in self.agent_chars if things[c].visible]
        for agent in agents:
            self.render_scope(agent, layers)

    def render_scope(self, agent, layers):
        pos_x, pos_y = agent.position
        off_x, off_y = self.scope_offset[agent.orientation]
        
        self.curtain[pos_x + off_x, pos_y + off_y] = True
        
        self.curtain[:, :] = np.logical_and(self.curtain, np.logical_not(layers[Sprites.WALL]))

                