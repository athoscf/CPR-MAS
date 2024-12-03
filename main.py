import numpy as np
import matplotlib.pyplot as plt
import gym
import random
from CommonsGame.resources import *
from CommonsGame.rl_models.q_learner import QLearner

# Parameters
numAgents = 1
epsilon = 0.1  # Exploration rate
alpha = 0.1    # Learning rate
gamma = 0.95  # Discount factor


# Environment setup
env = gym.make('CommonsGame:CommonsGame-v0', map_config=SmallMap, visual_radius=10, full_state=True)
action_space = env.action_space.n
obs_space_shape = (env.map_height + 2 * env.num_pad_pixels, env.map_width + 2 * env.num_pad_pixels, 3)  # Based on provided `getObservation` logic
agent_list = [QLearner(index, 1000000, env.action_space.n, learning_rate=alpha, discount_factor=gamma, exploration_rate=epsilon, exploration_decay=0.99) for index in range(numAgents)] # replace 10000 with the correct number of states


def calculate_metrics(episode_rewards, reward_times, tagged_steps, total_steps):
    U = np.mean(episode_rewards)  
    total_rewards = np.sum(episode_rewards)

    if total_rewards > 0:
        pairwise_diff_sum = sum(abs(x - y) for i, x in enumerate(episode_rewards) for y in episode_rewards[i+1:])
        E = 1 - (pairwise_diff_sum / (2 * numAgents * total_rewards))
    else:
        E = 0
    S = np.mean([np.mean(times) if times else 0 for times in reward_times])
    P = (total_steps - tagged_steps) / total_steps if total_steps > 0 else 0  

    efficiency.append(U)
    equality.append(E)
    sustainability.append(S)
    peace.append(P)

    return

def plot_metric(metric_values, metric_name):

    plt.figure(figsize=(10, 6))

    plt.plot(metric_values, label=metric_name)

    plt.xlabel('Episode')
    plt.ylabel(f'{metric_name}')
    plt.legend()
    plt.grid(True)

    plt.savefig(f'{metric_name}_history.png')
    plt.show()  
    return 


# Main loop
done = False

num_episodes = 15

efficiency = []
equality = []
sustainability = []
peace = []

for episode in range(num_episodes):
    done = False
    env.reset()

    episode_rewards = np.zeros(numAgents) 
    reward_times = [[] for _ in range(numAgents)] 
    tagged_steps = 0 
    total_steps = 0  

    # For each episode, run until done
    for t in range(1000):  # You can also set a maximum number of time steps per episode if desired
        if done:
            break

        # Get observations for all agents
        observations, done = env.get_observation()

        #if episode == 10:  # Example: Print observation of the first agent during episode 10
        #    print(observations[0])

        nActions = []
        
        for agent in agent_list:
            nActions.append(agent.select_action(observations[agent.id]))

        # Take a step in the environment
        nObservations, nRewards, nDone, nInfo = env.step(nActions)
        if len(nRewards) != numAgents:
            print(nRewards)

        if len(nObservations) != numAgents:
            print(nObservations)
        
        for agent in agent_list:
            agent.update_q_value(observations[agent.id], nActions[agent.id],
                                 nRewards[agent.id], nObservations[agent.id], nDone[agent.id])
        # Update metrics
            episode_rewards[agent.id] += nRewards[agent.id]
            if nRewards[agent.id] > 0:
                reward_times[agent.id].append(t)

        tagged_steps += nActions.count(7)
        total_steps += numAgents

    calculate_metrics(episode_rewards, reward_times, tagged_steps, total_steps)

    if episode:
        print(f"Episode {episode + 1}")


plot_metric(efficiency,'Efficiency(U)')
plot_metric(equality,'Equality(E)')
plot_metric(sustainability,'sustainability(S)')
plot_metric(peace,'Peace(P)')
