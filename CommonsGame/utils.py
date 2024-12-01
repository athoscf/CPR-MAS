import numpy as np
import random
from pycolab.rendering import ObservationToArray
from pycolab import ascii_art
from CommonsGame.constants import *
from CommonsGame.objects import *

def build_map(map_sketch, num_pad_pixels, agent_chars):
    num_agents = len(agent_chars)
    game_map = np.array(map_sketch)

    def padWith(vector, pad_width, iaxis, kwargs):
        del iaxis
        pad_value = kwargs.get('padder', ' ')
        vector[:pad_width[0]] = pad_value
        vector[-pad_width[1]:] = pad_value
        return vector

    # Put agents
    available_cells = np.argwhere(np.logical_and(game_map != Sprites.APPLE, game_map != Sprites.WALL))
    selected_cells = np.random.choice(available_cells.shape[0], size=(num_agents,), replace=False)
    agents_pos = available_cells[selected_cells, :]
    for idx, pos in enumerate(agents_pos):
        game_map[pos[0], pos[1]] = agent_chars[idx]
    
    # Put border walls
    game_map = np.pad(game_map, num_pad_pixels + 1, padWith, padder='=')

    game_map = [''.join(row.tolist()) for row in game_map]
    return game_map

def build_game(map_sketch, num_pad_pixels, agent_chars):
    game_map = build_map(map_sketch, num_pad_pixels, agent_chars)

    agentsOrder = list(agent_chars)
    random.shuffle(agentsOrder)
    
    return ascii_art.ascii_art_to_game(
        game_map,
        what_lies_beneath=' ',
        sprites=dict([(a, ascii_art.Partial(Agent, agent_chars)) for a in agent_chars]),
        drapes={Sprites.APPLE: ascii_art.Partial(Apple, agent_chars, num_pad_pixels),
                Sprites.SCOPE: ascii_art.Partial(Scope, agent_chars),
                Sprites.GIFT: ascii_art.Partial(Gift, agent_chars),
                Sprites.BEAM: ascii_art.Partial(Beam, agent_chars)},
        update_schedule=[Sprites.BEAM, Sprites.GIFT] + agentsOrder + [Sprites.SCOPE, Sprites.APPLE],
        z_order=[Sprites.SCOPE, Sprites.APPLE] + agentsOrder + [Sprites.BEAM, Sprites.GIFT]
    )

def rbg_to_char(observation):
    char_observation = []
    for row in observation:
        char_row = []
        for pixel in row:
            rgb_tuple = tuple(pixel)  
            char_row.append(COLOUR_TO_CHAR.get(rgb_tuple, ' ')) 
        char_observation.append(char_row)

    return char_observation

class ObservationToArrayWithRGB(object):
    def __init__(self, agent_chars):
        colour_mapping = dict([(agent, Colours.RED) for agent in agent_chars] + DEFAULT_COLOURS)

        # Rendering functions for the `board` representation and `RGB` values.
        self._renderers = {
            'RGB': ObservationToArray(value_mapping=colour_mapping)
        }

    def __call__(self, observation):
        # Perform observation rendering for agent and for video recording.
        result = {}
        for key, renderer in self._renderers.items():
            result[key] = renderer(observation)
        return result