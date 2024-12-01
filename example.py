import numpy as np
import gym
import warnings
from CommonsGame.constants import *
warnings.filterwarnings("ignore", category=FutureWarning)

numAgents = 2

env = gym.make('CommonsGame:CommonsGame-v0', num_agents=numAgents, visual_radius=5, map_sketch=small_map)
env.reset()
for t in range(100):
    nActions = [7, 7]
    nObservations, nRewards, nDone, nInfo = env.step(nActions)
    if (t == 99):
        print(nObservations)
    env.render()

