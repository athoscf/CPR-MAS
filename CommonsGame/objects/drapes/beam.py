import numpy as np
from pycolab import things as pythings
from CommonsGame.constants import *

class Beam(pythings.Drape):
    
    def __init__(self, curtain, character, agent_chars):
        super().__init__(curtain, character)
        self.agent_chars = agent_chars
        self.render_beam = {
            Orientations.NORTH: self.render_north,
            Orientations.EAST: self.render_east,
            Orientations.SOUTH: self.render_south,
            Orientations.WEST: self.render_west 
        }
        
    def update(self, actions, board, layers, backdrop, things, the_plot):
        if actions is None: return
        
        np.logical_and(self.curtain, False, self.curtain)
        
        for agent in self.tagging_agents(actions, things):
            x_min, x_max, y_min, y_max = self.render_beam[agent.orientation](agent.position, layers, BeamRange.WIDTH, BeamRange.HEIGHT)
            self.curtain[x_min:x_max, y_min:y_max] = True

    def tagging_agents(self, actions, things):
        return [ 
            things[self.agent_chars[i]] 
            for i, a in enumerate(actions) 
            if a == Actions.TAG and things[self.agent_chars[i]].visible
        ]
        
    def render_north(self, pos, layers, width, height):
        if np.any(layers[Sprites.WALL][pos[0] - height:pos[0], pos[1] - width:pos[1] + width + 1]):
            collision_idxs = np.argwhere(layers[Sprites.WALL][pos[0] - height:pos[0], pos[1] - width:pos[1] + width + 1])
            height -= (np.max(collision_idxs) + 1)
            
        return pos[0] - height, pos[0], pos[1] - width, pos[1] + width + 1
        
    def render_east(self, pos, layers, width, height):
        if np.any(layers[Sprites.WALL][pos[0] - width:pos[0] + width + 1, pos[1] + 1:pos[1] + height + 1]):
            collision_idxs = np.argwhere(layers[Sprites.WALL][pos[0] - width:pos[0] + width + 1, pos[1] + 1:pos[1] + height + 1])
            height = np.min(collision_idxs)
            
        return pos[0] - width, pos[0] + width + 1, pos[1] + 1, pos[1] + height + 1
        
    def render_south(self, pos, layers, width, height):
        if np.any(layers[Sprites.WALL][pos[0] + 1:pos[0] + height + 1, pos[1] - width:pos[1] + width + 1]):
            collision_idxs = np.argwhere(layers[Sprites.WALL][pos[0] + 1:pos[0] + height + 1, pos[1] - width:pos[1] + width + 1])
            height = np.min(collision_idxs)
            
        return pos[0] + 1, pos[0] + height + 1, pos[1] - width, pos[1] + width + 1
        
    def render_west(self, pos, layers, width, height):
        if np.any(layers[Sprites.WALL][pos[0] - width:pos[0] + width + 1, pos[1] - height:pos[1]]):
            collision_idxs = np.argwhere(layers[Sprites.WALL][pos[0] - width:pos[0] + width + 1, pos[1] - height:pos[1]])
            height -= (np.max(collision_idxs) + 1)
            
        return pos[0] - width, pos[0] + width + 1, pos[1] - height, pos[1]
        


