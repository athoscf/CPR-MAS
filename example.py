import numpy as np
import gym
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

numAgents = 11

env = gym.make('CommonsGame:CommonsGame-v0', num_agents=numAgents, visual_radius=4)
env.reset()
for t in range(50):
    nActions = np.random.randint(low=0, high=env.action_space.n, size=(numAgents,)).tolist()
    nObservations, nRewards, nDone, nInfo = env.step(nActions)
    env.render()
    if t == 9:
        print(nObservations[0])