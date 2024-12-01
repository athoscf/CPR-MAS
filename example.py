import numpy as np
import gym
import warnings
from CommonsGame.constants import *
warnings.filterwarnings("ignore", category=FutureWarning)

numAgents = 3

env = gym.make('CommonsGame:CommonsGame-v0', num_agents=numAgents, visual_radius=5, map_sketch=small_map)
env.reset()

# while i dont click esc
for t in range(3):
    nActions = [8, 6, 6]
    nObservations, nRewards, nDone, nInfo = env.step(nActions)
    env.render()

