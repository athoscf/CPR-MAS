import gym
import numpy as np
from CommonsGame.model.agent import Agent
from CommonsGame.model.replay_buffer import ReplayBuffer as ReplayBuffer
from CommonsGame.resources import *
from CommonsGame.model.metrics import *

class TestModel():
    
    def __init__(self, num_episodes=5000, map=SmallMap, visual_radius=5, warmup_steps=1000):
        self.num_episodes = num_episodes
        self.num_agents = map.num_agents
        self.visual_radius = visual_radius
        self.warmup_steps = warmup_steps

        self.input_dims = [visual_radius * 2 + 1, visual_radius * 2 + 1, 3]
        self.env = gym.make('CommonsGame:CommonsGame-v0', map_config=map, visual_radius=visual_radius)

        self.agents = [Agent(input_dims=self.input_dims, num_actions=8) for _ in range(self.num_agents)]

    def execute(self):
        print("warming up replay buffer...")
        self.warmup_replay_buffer()
        print("replay buffer warmed up...")

        scores, eps_history, metrics_values, loss_history = [], [], [], []

        for episode in range(1, self.num_episodes + 1):
            self.run_episode(episode, scores, eps_history, loss_history,metrics_values)
        plot_metrics(metrics_values,self.num_episodes,'result.png')
    def warmup_replay_buffer(self):
        step = 0 
        while step < self.warmup_steps:
            done = [False]
            observations = self.env.reset()
            while not done[0]:
                actions = self.choose_random_actions(observations)

                observations_, rewards, done, info = self.env.step(actions)

                self.store_transitions(observations, observations_, actions, rewards, done)
         
                observations = observations_
                step += 1
                if step > self.warmup_steps:
                    break
        self.env.reset()

    def train_agent(self, agent, observation, observation_, action, reward, done):
        if type(observation_) == type(None):
            return
        if type(observation) == type(None):
            observation = np.copy(agent.replay_buffer.buffer[-1][0])

        agent.store_transition(observation, action, reward, observation_, done)
        loss = agent.learn()
        return loss

    def train_agents(self, observations, losses, observations_, actions, rewards, done):    
        for i, agent in enumerate(self.agents):
            loss = self.train_agent(agent, observations[i], observations_[i], actions[i], rewards[i], done[i])
            if type(loss) != type(None):
                losses.append(loss)
        return losses

    def choose_random_actions(self, observations):
        actions = []
        for i, agent in enumerate(self.agents):
            if type(observations[i]) == type(None):
                action = Actions.STAND_STILL
            else:
                action = np.random.choice(agent.action_space)
            actions.append(action)
        return actions

    def choose_actions(self, observations):
        actions = []
        for i, agent in enumerate(self.agents):
            action = agent.choose_action(observations[i])
            actions.append(action)
        return actions

    def store_transitions(self, observations, observations_, actions, rewards, done):
        for i, agent in enumerate(self.agents):
            if type(observations_[i]) == type(None):
                continue
            if type(observations[i]) == type(None):
                observations[i] = np.copy(agent.replay_buffer.buffer[-1][0])
            agent.store_transition(observations[i], actions[i], rewards[i], observations_[i], done[i])

    def run_episode(self, episode, scores, eps_history, loss_history, metrics_values):
        metrics = Metrics(self.num_agents)
        score = 0
        done = [False]
        observations = self.env.reset()
        losses = []
        step = 0
        while not done[0] and step < 1000:
            # Each agent chooses its action.
            actions = self.choose_actions(observations) 
            
            # Actions are played, rewards are received.
            new_observations, rewards, done, info = self.env.step(actions)
            metrics.add_step(new_observations,rewards)
            score += rewards[0]

            losses = self.train_agents(observations, losses, new_observations, actions, rewards, done)

            observations = new_observations
            step += 1
            
        for agent in self.agents:
            agent.decay_epsilon()

        # Save scores
        metrics.calculate_metrics()
        metrics_values.append(metrics)
        scores.append(score)
        eps_history.append(self.agents[0].epsilon)
        loss_history.append(np.array(losses).mean())

        avg_score = np.mean(scores[-100:])

        print('Episode {} Score: {:.2f} Average Score: {:.2f} Epsilon {:.2f}'.format(episode, scores[-1], avg_score, self.agents[0].epsilon))
        