import random
import numpy as np
import matplotlib.pyplot as plt
import gym
from gym import spaces
from pycolab import ascii_art
from CommonsGame.constants import *
from CommonsGame.utils import buildMap, ObservationToArrayWithRGB
from CommonsGame.objects import *

class CommonsGame(gym.Env):

    metadata = {
        'render_modes': ['None', 'human'],
        'render_fps': 5    
    }

    def __init__(self, numAgents, visualRadius, mapSketch=bigMap, fullState=False):
        super(CommonsGame, self).__init__()
        self.fullState = fullState
        # Setup spaces
        self.action_space = spaces.Discrete(NUM_ACTIONS)
        obHeight = obWidth = visualRadius * 2 + 1
        # Setup game
        self.numAgents = numAgents
        self.sightRadius = visualRadius
        self.agentChars = agentChars = Sprites.AGENTS[0:numAgents]
        self.mapHeight = len(mapSketch)
        self.mapWidth = len(mapSketch[0])
        if fullState:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.mapHeight + 2, self.mapWidth + 2, 3),
                                                dtype=np.uint8)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(obHeight, obWidth, 3), dtype=np.uint8)
        self.numPadPixels = numPadPixels = visualRadius - 1
        self.gameField = buildMap(mapSketch, numPadPixels=numPadPixels, agentChars=agentChars)
        self.state = None
        # Pycolab related setup:        
        self._game = self.buildGame()
        colourMap = dict([(agent, Colors.RED) for agent in agentChars]  # Agents
                         + [(Sprites.WALL, Colors.WHITE)]  # Steel Impassable wall
                         + [(Sprites.EMPTY, Colors.BLACK)]  # Black background
                         + [(Sprites.APPLE, Colors.GREEN)]  # Green Apples
                         + [(Sprites.BEAM, Colors.YELLOW)]  # Yellow beam
                         + [(Sprites.GIFT, Colors.BLUE)]  # Blue gift
                         + [(Sprites.SCOPE, Colors.GRAY)])  # Grey scope
        self.obToImage = ObservationToArrayWithRGB(colour_mapping=colourMap)

    def buildGame(self):
        agentsOrder = list(self.agentChars)
        random.shuffle(agentsOrder)
        return ascii_art.ascii_art_to_game(
            self.gameField,
            what_lies_beneath=' ',
            sprites=dict(
                [(a, ascii_art.Partial(Agent, self.agentChars)) for a in self.agentChars]),
            drapes={Sprites.APPLE: ascii_art.Partial(Apple, self.agentChars, self.numPadPixels),
                    Sprites.SCOPE: ascii_art.Partial(Scope, self.agentChars),
                    Sprites.GIFT: ascii_art.Partial(Gift, self.agentChars),
                    Sprites.BEAM: ascii_art.Partial(Beam, self.agentChars)},
            update_schedule=[Sprites.BEAM, Sprites.GIFT] + agentsOrder + [Sprites.SCOPE, Sprites.APPLE],
            z_order=[Sprites.SCOPE, Sprites.APPLE] + agentsOrder + [Sprites.BEAM, Sprites.GIFT]
        )

    def step(self, nActions):
        nInfo = {'n': []}
        self.state, nRewards, _ = self._game.play(nActions)
        nObservations, done = self.getObservation()
        nDone = [done] * self.numAgents
        return nObservations, nRewards, nDone, nInfo

    def reset(self):
        # Reset the state of the environment to an initial state
        self._game = self.buildGame()
        self.state, _, _ = self._game.its_showtime()
        nObservations, _ = self.getObservation()
        return nObservations

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        board = self.obToImage(self.state)['RGB'].transpose([1, 2, 0])
        board = board[self.numPadPixels:self.numPadPixels + self.mapHeight + 2,
                self.numPadPixels:self.numPadPixels + self.mapWidth + 2, :]
        plt.figure(1)
        plt.imshow(board)
        plt.axis("off")
        plt.show(block=False)
        # plt.show()
        plt.pause(.05)
        plt.clf()

    def getObservation(self):
        done = not (np.logical_or.reduce(self.state.layers['@'], axis=None))
        agents = [self._game.things[c] for c in self.agentChars]
        observations = []
        board = self.obToImage(self.state)['RGB'].transpose([1, 2, 0])
        for agent in agents:
            observation = self.getAgentObservation(board, agent)
            observations.append(observation)
        return observations, done

    def getAgentObservation(self, board, agent):
        if agent.visible or agent.timeout == 25:
            if self.fullState:
                observation = np.copy(board)
                if agent.visible:
                    observation[agent.position[0], agent.position[1], :] = [0, 0, 255]
                observation = observation[self.numPadPixels:self.numPadPixels + self.mapHeight + 2,
                        self.numPadPixels:self.numPadPixels + self.mapWidth + 2, :]
            else:
                observation = np.copy(board[
                                agent.position[0] - self.sightRadius:agent.position[0] + self.sightRadius + 1,
                                agent.position[1] - self.sightRadius:agent.position[1] + self.sightRadius + 1, :])
                if agent.visible:
                    observation[self.sightRadius, self.sightRadius, :] = [0, 0, 255]
            observation = observation / 255.0
        else:
            observation = None
        return observation

