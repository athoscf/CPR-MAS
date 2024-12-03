import numpy as np

class Metrics:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.observations = []
        self.rewards = []

        self.efficiency = None
        self.equality = None
        self.sustainability = None
        self.peace = None

    def add_step(self, observation, rewards):
        self.observations.append(observation)
        self.rewards.append(rewards)
    
    def calculate_efficiency(self):
        sm = sum(sum(agent_rewards) for agent_rewards in self.rewards)
        self.efficiency = sm / self.num_agents

    def calculate_equality(self):
        total_rewards = np.array(self.rewards)
        episode_rewards = np.sum(total_rewards,axis = 0)
        gini = 0
        for i,j in enumerate(1,episode_rewards[1:]):
            gini += np.sum(np.abs(i-episode_rewards[j]))
        
        return gini / len(episode_rewards)**2 * 