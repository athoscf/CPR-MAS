import numpy as np
from pycolab import things as pythings
from CommonsGame.resources.constants import *

class Gift(pythings.Drape):
    
    def __init__(self, curtain, character, agent_chars):
        super().__init__(curtain, character)
        self.agent_chars = agent_chars
        self.render_gift = {
            Orientations.NORTH: self.render_north,
            Orientations.EAST: self.render_east,
            Orientations.SOUTH: self.render_south,
            Orientations.WEST: self.render_west 
        }
        
    def update(self, actions, board, layers, backdrop, things, the_plot):
        if actions is None: return
        
        np.logical_and(self.curtain, False, self.curtain)
        
        rewards = [0 for i in range(len(self.agent_chars))]
        
        for agent in self.gifting_agents(actions, things):
            x_min, x_max, y_min, y_max = self.render_gift[agent.orientation](agent.position, layers, GiftRange.WIDTH, GiftRange.HEIGHT)
            self.curtain[x_min:x_max, y_min:y_max] = True
            
            gifted_agents = self.gifted_agents(agent.position, x_min, x_max, y_min, y_max, things)
            if len(gifted_agents) > 0:
                tmp_rewards = self.gift_agents(agent, gifted_agents)
                rewards = [r + tr for r, tr in zip(rewards, tmp_rewards)]
                
        the_plot.add_reward(rewards)

    def gifting_agents(self, actions, things):
        return [ 
            things[self.agent_chars[i]] 
            for i, a in enumerate(actions) 
            if a == Actions.GIFT and things[self.agent_chars[i]].visible and things[self.agent_chars[i]].reward > 0
        ]
        
    def gifted_agents(self, pos, x_min, x_max, y_min, y_max, things):
        gifted_agents = []
        
        for c in self.agent_chars:
            agent = things[c]
            if agent.visible:
                x, y = agent.position
                if x_min <= x < x_max and y_min <= y < y_max:
                    gifted_agents.append(agent)

        return gifted_agents
        
    def gift_agents(self, gifting_agent, gifted_agents):
        rewards = [0 for i in range(len(self.agent_chars))]
        gift_weight = 1 / len(gifted_agents)
        
        gifting_agent.reward -= 1
        rewards[gifting_agent.index] -= 1
        
        for agent in gifted_agents:
            agent.reward += gift_weight
            rewards[agent.index] += gift_weight
        
        return rewards
        
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
