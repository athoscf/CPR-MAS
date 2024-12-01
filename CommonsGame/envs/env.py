import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from CommonsGame.constants import *
from CommonsGame.utils import *
from CommonsGame.objects import *

class CommonsGame(gym.Env):

    metadata = {
        'render_modes': ['None', 'human'],
        'render_fps': 5    
    }

    def __init__(self, num_agents, visual_radius, map_sketch=big_map, full_state=False):
        super(CommonsGame, self).__init__()
        self.full_state = full_state
        
        # Setup spaces
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        ob_height = ob_width = visual_radius * 2 + 1
        
        # Setup game
        self.num_agents = num_agents
        self.sight_radius = visual_radius
        self.agent_chars = agent_chars = Sprites.AGENTS[0:num_agents]
        self.map_height = len(map_sketch)
        self.map_width = len(map_sketch[0])
        self.map_sketch = map_sketch
        
        if full_state:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.map_height + 2, self.map_width + 2, 3), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(ob_height, ob_width, 3), dtype=np.uint8)
        
        self.num_pad_pixels = num_pad_pixels = visual_radius - 1
        self.game = build_game(map_sketch, num_pad_pixels, agent_chars)        
        self.state = None
        
        # Pycolab related setup:        
        self.ob_to_image = ObservationToArrayWithRGB(agent_chars)

    def step(self, actions):
        nInfo = {'n': []}
        self.state, rewards, _ = self.game.play(actions)
        observations, done = self.get_observation()
        nDone = [done] * self.num_agents
        return observations, rewards, nDone, nInfo

    def reset(self):
        # Reset the state of the environment to an initial state
        self.game = build_game(self.map_sketch, self.num_pad_pixels, self.agent_chars)  
        self.state, _, _ = self.game.its_showtime()
        observations, _ = self.get_observation()
        return observations

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        board = self.ob_to_image(self.state)['RGB'].transpose([1, 2, 0])
        board = board[self.num_pad_pixels:self.num_pad_pixels + self.map_height + 2,
                self.num_pad_pixels:self.num_pad_pixels + self.map_width + 2, :]
        plt.figure(1)
        plt.imshow(board)
        plt.axis("off")
        plt.show(block=False)
        plt.pause(.05)
        plt.clf()

    def get_observation(self):
        done = not (np.logical_or.reduce(self.state.layers[Sprites.APPLE], axis=None))
        
        observations = []
        board = self.ob_to_image(self.state)['RGB'].transpose([1, 2, 0])
        for agent in [self.game.things[c] for c in self.agent_chars]:
            observation = self.get_agent_observation(board, agent)
            observations.append(observation)
        
        return observations, done

    def get_agent_observation(self, board, agent):
        if not agent.visible and agent.timeout != TIMEOUT: return None
       
        if self.full_state:
            observation = np.copy(board)
            observation[agent.position[0], agent.position[1], :] = Colours.RED
            observation = observation[self.num_pad_pixels:self.num_pad_pixels + self.map_height + 2,
                    self.num_pad_pixels:self.num_pad_pixels + self.map_width + 2, :]
        else:
            observation = np.copy(board[
                            agent.position[0] - self.sight_radius:agent.position[0] + self.sight_radius + 1,
                            agent.position[1] - self.sight_radius:agent.position[1] + self.sight_radius + 1, :])
            observation[self.sight_radius, self.sight_radius, :] = Colours.RED

        return rbg_to_char(observation)
