import numpy as np
import matplotlib.pyplot as plt

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
        self.efficiency = np.mean( [sum(sublist[i] for sublist in self.rewards) for i in range(len(self.rewards[0]))])

    def calculate_equality(self):
        total_rewards = np.array(self.rewards)
        episode_rewards = np.sum(total_rewards, axis=0)
        
        abs_differences = np.abs(episode_rewards[:, None] - episode_rewards)
        gini = np.sum(abs_differences)

        n = len(episode_rewards)

        if (n == 0 or sum(episode_rewards) == 0):
            self.equality = 0
            return
        gini_coefficient = gini / (2 * n * sum(episode_rewards))
        self.equality = 1 - gini_coefficient
    
    def calculate_sustainability(self):
        t = 0
        for i in range(len(self.rewards)):
            ts = np.array(self.rewards[i]) * i
            sum_times = np.sum(ts)
            t += sum_times

        self.sustainability = t / len(self.rewards)

    def calculate_peace(self):
        steps = len(self.observations)
        tagged = 0
        for observation_step in self.observations:
            tagged += sum([1 for obs in observation_step if obs is None])

        total_observations = steps * self.num_agents
        self.peace = (total_observations - tagged) / steps

    def calculate_metrics(self):
        self.calculate_efficiency()
        self.calculate_equality()
        self.calculate_sustainability()
        self.calculate_peace()

        self.observations = []
        self.rewards = []

def plot_metrics(metrics_values, num_episodes, filename): 
    fig, ax = plt.subplots(4, 1, sharex=True, figsize=(6, 10))

    if not isinstance(metrics_values, list):
        metrics_values = [metrics_values]

    efficiency = [m.efficiency for m in metrics_values]
    equality = [m.equality for m in metrics_values]
    sustainability = [m.sustainability for m in metrics_values]
    peace = [m.peace for m in metrics_values]

    x = np.arange(1, num_episodes + 1)  

    window_size = 100
    min_periods = 25

    # Loop through the metrics
    for i, metric in enumerate([efficiency, sustainability, equality, peace]):
        metric = np.array(metric)

        # Compute rolling average
        rolling_avg = np.array([
            np.mean(metric[max(0, j - window_size + 1):j + 1])
            if j + 1 >= min_periods else np.nan
            for j in range(len(metric))
        ])

        # Plot metrics
        ax[i].plot(x, metric, alpha=0.5, label="Value")
        ax[i].plot(x, rolling_avg, label="Rolling Avg", linestyle='--')

        ax[i].legend()

    # Set axis labels
    ax[0].set_ylabel('Efficiency (U)', fontsize=14)
    ax[1].set_ylabel('Sustainability (S)', fontsize=14)
    ax[2].set_ylabel('Equality (E)', fontsize=14)
    ax[3].set_ylabel('Peacefulness (P)', fontsize=14)
    ax[3].set_xlabel('Episode', fontsize=14)

    # Add grid lines
    for i in range(4):
        ax[i].yaxis.grid(linestyle='--')

    fig.tight_layout()
    fig.savefig(filename)
    print("Plotted metrics!")