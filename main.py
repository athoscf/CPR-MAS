import numpy as np
import matplotlib.pyplot as plt
import gym
import random
from CommonsGame.resources import *
from libs.metrics import *
# Parameters
numAgents = 9
epsilon = 0.05  # Exploration rate
alpha = 0.1    # Learning rate
gamma = 0.99   # Discount factor

# Environment setup
env = gym.make('CommonsGame:CommonsGame-v0', map_config=SmallMap, visual_radius=4, full_state=True)
action_space = env.action_space.n
obs_space_shape = (env.map_height + 2 * env.num_pad_pixels, env.map_width + 2 * env.num_pad_pixels, 3)  # Based on provided `getObservation` logic
Q_tables = [np.zeros((10000, action_space)) for _ in range(numAgents)]  # Replace 10000 with an appropriate state space size

# Helper functions
def state_to_index(state):
    """Convert observation (state) to a hashable index."""
    if state is None:
        return -1  # Handle None state
    return hash(state) % 10000  # Simplified; modify if needed for large state spaces

def select_action(agent_id, observation):
    """Select an action using epsilon-greedy policy."""
    state_idx = state_to_index(observation)
    if state_idx == -1 or random.uniform(0, 1) < epsilon:
        return np.random.randint(0, action_space)  # Exploration
    return np.argmax(Q_tables[agent_id][state_idx])  # Exploitation

def update_q_table(agent_id, obs, action, reward, next_obs):
    """Update Q-table using the Bellman equation."""
    state_idx = state_to_index(obs)
    next_state_idx = state_to_index(next_obs)
    if state_idx == -1 or next_state_idx == -1:  # Skip updates for invalid states
        return
    best_next_action = np.argmax(Q_tables[agent_id][next_state_idx])
    td_target = reward + gamma * Q_tables[agent_id][next_state_idx][best_next_action]
    td_error = td_target - Q_tables[agent_id][state_idx][action]
    Q_tables[agent_id][state_idx][action] += alpha * td_error

# Main loop
done = False

num_episodes = 1000

metrics_values = []
for episode in range(1,num_episodes+1):
    done = False
    env.reset()
    # For each episode, run until done
    for t in range(1000):  # You can also set a maximum number of time steps per episode if desired
        if done:
            break
        metrics = Metrics(numAgents)
        # Get observations for all agents
        observations, done = env.get_observation()

        #if episode == 10:  # Example: Print observation of the first agent during episode 10
        #    print(observations[0])

        nActions = []

        # Determine actions for each agent
        for agent_id in range(numAgents):
            action = select_action(agent_id, observations[agent_id])
            nActions.append(action)

        # Take a step in the environment
        nObservations, nRewards, nDone, nInfo = env.step(nActions)
        metrics.add_step(nObservations,nRewards)

        if len(nRewards) != numAgents:
            print(nRewards)

        if len(nObservations) != numAgents:
            print(nObservations)
        
        # Update Q-tables for each agent
        for agent_id in range(numAgents):
            update_q_table(agent_id, observations[agent_id], nActions[agent_id],
                           nRewards[agent_id], nObservations[agent_id])
            
    metrics.calculate_metrics()
    metrics_values.append(metrics)
    print(f"Episode {episode}")


plot_metrics(metrics_values,num_episodes,'results.png')