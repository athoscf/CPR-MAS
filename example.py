import numpy as np
import gym
import warnings
from CommonsGame import *

warnings.filterwarnings("ignore", category=FutureWarning)

numAgents = 1

env = gym.make('CommonsGame:CommonsGame-v0', map_config=UnequalEntranceMap, visual_radius=4)
env.reset()\
    
    
inp = ''
# for t in range(50):
#     nActions = np.random.randint(low=0, high=env.action_space.n, size=(numAgents,)).tolist()
#     nObservations, nRewards, nDone, nInfo = env.step(nActions)
#     env.render()
#     if t == 9:
#         print(nObservations[0])

while inp != 'q':
    env.render()
    inp = input('Press q to exit: ')