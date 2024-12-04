import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from CommonsGame.resources.constants import *
from CommonsGame.envs.utils import *
from CommonsGame.objects import *

class CommonsGame(gym.Env):

    metadata = {
        'render_modes': ['None', 'human'],
        'render_fps': 5    
    }

    def __init__(self, map_config, visual_radius, full_state=False):
        super(CommonsGame, self).__init__()
        self.full_state = full_state
        
        # Setup spaces
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        ob_height = ob_width = visual_radius * 2 + 1
        
        # Setup game
        self.num_agents = map_config.num_agents
        self.sight_radius = visual_radius
        self.agent_chars = agent_chars = map_config.agent_chars
        self.map_height = len(map_config.map)
        self.map_width = len(map_config.map[0])
        self.map = map_config.map
        
        if full_state:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.map_height + 2, self.map_width + 2, 3), dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(ob_height, ob_width, 3), dtype=np.uint8)
        
        self.num_pad_pixels = num_pad_pixels = visual_radius - 1
        self.game = build_game(self.map, num_pad_pixels, agent_chars)        
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
        self.game = build_game(self.map, self.num_pad_pixels, self.agent_chars)  
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
        board = self.ob_to_image(self.state)['RGB'].transpose([1, 2, 0])

        observations = []
        for agent in [self.game.things[c] for c in self.agent_chars]:
            observation = self.get_agent_observation(board, agent)
            observations.append(observation)
        
        return observations, done

    def get_agent_observation(self, board, agent):
        # If the agent is not visible and has timed out, return a zero-padded observation
        if not agent.visible and agent.timeout != TIMEOUT:
            if self.full_state:
                observation = np.zeros((self.map_height + 2, self.map_width + 2, board.shape[-1]), dtype=np.float32)
            else:
                observation = np.zeros((2 * self.sight_radius + 1, 2 * self.sight_radius + 1, board.shape[-1]), dtype=np.float32)
            return observation

        # Full state observation
        if self.full_state:
            observation = np.copy(board)
            observation[agent.position[0], agent.position[1], :] = Colours.PURPLE
            observation = observation[self.num_pad_pixels:self.num_pad_pixels + self.map_height + 2,
                                    self.num_pad_pixels:self.num_pad_pixels + self.map_width + 2, :]
        # Partial state observation
        else:
            observation = np.copy(board[
                agent.position[0] - self.sight_radius:agent.position[0] + self.sight_radius + 1,
                agent.position[1] - self.sight_radius:agent.position[1] + self.sight_radius + 1, :])
            observation[self.sight_radius, self.sight_radius, :] = Colours.PURPLE

        return observation

