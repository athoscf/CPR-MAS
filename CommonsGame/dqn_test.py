import logging
import os

from rl_models.dqn_agent import Agent
import gym
import numpy as np
from resources import *
from libs.metrics import *
import threading



def warmup_replay_buffer(env, warmup_steps, agents):
    step = 0
    
    while step < warmup_steps:
        done = [False]
        observations = env.reset()
        while not done[0]:

            # Each agent chooses its action.
            actions = []
            for ag in range(len(agents)):
                observation = observations[ag]
                agent = agents[ag]
                if type(observation) == type(None):
                    action = Actions.STAND_STILL
                else:
                    action = np.random.choice(agent.action_space)
                actions.append(action)

            # Actions are played, rewards are received.
            observations_, rewards, done, info = env.step(actions)
            
            for ag in range(len(agents)):
                if type(observations_[ag]) == type(None):
                    continue
                if type(observations[ag]) == type(None):
                    observations[ag] = np.copy(agents[ag].ER.buffer[-1][0])

                agents[ag].store_transition(observations[ag], actions[ag], rewards[ag], observations_[ag], done[ag])

            observations = observations_
            step += 1
            if step > warmup_steps:
                break

    env.reset()

def train_agent(agent, observation, observation_, action, reward, done):
    if type(observation_) == type(None):
        return
    if type(observation) == type(None):
        observation = np.copy(agent.ER.buffer[-1][0])

    agent.store_transition(observation, action, reward, observation_, done)
    loss = agent.learn()
    return loss

def train_agents(observations, agents, losses, observations_, actions, rewards, done):    
    for i, agent in enumerate(agents):
        loss = train_agent(agent, observations[i], observations_[i], actions[i], rewards[i], done[i])
        if type(loss) != type(None):
            losses.append(loss)
    return losses

def choose_actions(observations, agents):
    actions = []
    for i, agent in enumerate(agents):
        action = agent.choose_action(observations[i])
        actions.append(action)
    return actions

def run_episode(episode, env, agents, scores, eps_history, loss_history, metrics_value):

    
    score = 0
    done = [False]
    observations = env.reset()
    metrics = Metrics(len(agents))
    losses = []
    step = 0
    while not done[0] and step < 1000:
        # Each agent chooses its action.
        actions = choose_actions(observations, agents) 
        
        # Actions are played, rewards are received.
        new_observations, rewards, done, info = env.step(actions)
        #social_metrics.add_step(new_observations, rewards)
        metrics.add_step(new_observations,rewards)
        score += rewards[0]

        losses = train_agents(observations, agents, losses, new_observations, actions, rewards, done)

        # Current observations will be next old observations
        observations = new_observations
        step += 1
    for agent in agents:
        agent.epsilon_decay()

    # Save scores
    metrics.calculate_metrics()
    metrics_value.append(metrics)
    scores.append(score)
    eps_history.append(agents[0].epsilon)
    loss_history.append(np.array(losses).mean())

    avg_score = np.mean(scores[-100:])

    print('Episode {} Score: {:.2f} Average Score: {:.2f} Epsilon {:.2f}'.format(episode, scores[-1], avg_score, agents[0].epsilon))
            

def main():    
    
    # Hyperparameters
    n_episodes = 15
    num_agents = SmallMap.num_agents
    visual_radius = 5
    warmup_steps = 50000

    input_dims = [visual_radius * 2 + 1, visual_radius * 2 + 1, 3]
    env = gym.make('CommonsGame:CommonsGame-v0', map_config=SmallMap, visual_radius=visual_radius)

    agents = [Agent(input_dims=input_dims, n_actions=8) for _ in range(num_agents)]

    print("warming up replay buffer...")
    warmup_replay_buffer(env, warmup_steps, agents)
    print("replay buffer warmed up...")

    scores, eps_history, metrics_values , loss_history = [], [], [], []

    for episode in range(1, n_episodes + 1):
        run_episode(episode, env, agents, scores, eps_history, loss_history, metrics_values)
    plot_metrics(metrics_values,episode,'results.png')


     
if __name__ == "__main__":
    main()