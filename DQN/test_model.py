import gym
import numpy as np
import threading
import matplotlib
from matplotlib import animation
from DQN.agent import Agent
from DQN.replay_buffer import ReplayBuffer as ReplayBuffer
from CommonsGame.resources import *
from DQN.metrics import *

matplotlib.use('Agg')

class TestModel():
    
    def __init__(self, map=SmallMap, action_policy=ActionPolicies.TAG_AND_GIFT, num_episodes=10, visual_radius=5, warmup_steps=5000): 
        self.num_episodes = num_episodes
        self.num_agents = map.num_agents
        self.visual_radius = visual_radius
        self.warmup_steps = warmup_steps
        self.map = map
        self.empty_board = np.zeros((2 * visual_radius + 1, 2 * visual_radius + 1, 3), dtype=np.float32)
        self.action_policy = action_policy
        
        self.input_dims = [visual_radius * 2 + 1, visual_radius * 2 + 1, 3]
        self.env = gym.make('CommonsGame:CommonsGame-v0', map_config=map, visual_radius=visual_radius)

        self.agents = self.create_agents(action_policy)

    def execute(self):
        self.warmup_replay_buffer()
        metrics_values = []
        for episode in range(1, self.num_episodes + 1):
            self.run_episode(episode, metrics_values)
            if episode % 100 == 0: 
                Metrics.plot(metrics_values, episode, self.map, self.action_policy)
        
    def create_agents(self, action_policy):
        # create csv file for agents' metrics
        self.agent_filename = FILE_PATHS[self.map] + self.action_policy + f"/agent_metrics"
        
        if action_policy == ActionPolicies.TAG_AND_GIFT:
            agent_list = []
            rows = ""
            for i in range(self.num_agents):
                rows += f"agent{i+1},"
                agent = Agent(input_dims=self.input_dims, tag_enabled=True, gift_enabled=True)
                agent_list.append(agent)
            self.agent_filename += "_tag_and_gift.csv"
            # create csv file
            with open(self.agent_filename, "w") as f:
                f.write("episode," + rows + "\n")
            return agent_list
        elif action_policy == ActionPolicies.TAG_ONLY:
            agent_list = []
            rows = ""
            for i in range(self.num_agents):
                rows += f"agent{i+1},"
                agent = Agent(input_dims=self.input_dims, tag_enabled=True, gift_enabled=False)
                agent_list.append(agent)
            self.agent_filename += "_tag_only.csv"
            # create csv file
            with open(self.agent_filename, "w") as f:
                f.write("episode," + rows + "\n")
            return agent_list
        elif action_policy == ActionPolicies.GIFT_ONLY:
            agent_list = []
            rows = ""
            for i in range(self.num_agents):
                rows += f"agent{i+1},"
                agent = Agent(input_dims=self.input_dims, tag_enabled=False, gift_enabled=True)
                agent_list.append(agent)
            self.agent_filename += "_gift_only.csv"
            # create csv file
            with open(self.agent_filename, "w") as f:
                f.write("episode," + rows + "\n")
            return agent_list
        elif action_policy == ActionPolicies.MIXED:
            agent_list = []
            rows = ""
            half = self.num_agents//2
            for i in range(half):
                rows += f"gift_agent{i+1},"
                agent = Agent(input_dims=self.input_dims, tag_enabled=False, gift_enabled=True)
                agent_list.append(agent)
            for i in range(half):
                rows += f"tag_agent{i+half+1},"
                agent = Agent(input_dims=self.input_dims, tag_enabled=True, gift_enabled=False)
                agent_list.append(agent)
            self.agent_filename += "_mixed.csv"
            # create csv file
            with open(self.agent_filename, "w") as f:
                f.write("episode," + rows + "\n")
            return agent_list 
        elif action_policy == ActionPolicies.DEFAULT:
            agent_list = []
            rows = ""
            for i in range(self.num_agents):
                rows += f"agent{i+1},"
                agent = Agent(input_dims=self.input_dims, tag_enabled=False, gift_enabled=False)
                agent_list.append(agent)
            self.agent_filename += "_default.csv"
            # create csv file
            with open(self.agent_filename, "w") as f:
                f.write("episode," + rows + "\n")
            return agent_list
        
    def warmup_replay_buffer(self):
        print("Warming up replay buffer")
        done = [False]
        step = 0
        observations = self.env.reset()
        while step < self.warmup_steps:
            actions = self.choose_random_actions(observations)
            observations_, rewards, done, info = self.env.step(actions)
            self.store_transitions(observations, observations_, actions, rewards, done)
            observations = observations_
            step += 1
        self.env.reset()
        print("Replay buffer warmed up")

    def train_agent(self, agent, observation, observation_, action, reward, done):
        if np.array_equal(observation_, self.empty_board):
            return
        if np.array_equal(observation, self.empty_board):
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
            if np.array_equal(observations[i], self.empty_board):
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
            if np.array_equal(observations_[i], self.empty_board):
                continue
            if np.array_equal(observations[i], self.empty_board):
                observations[i] = np.copy(agent.replay_buffer.buffer[-1][0])
            agent.store_transition(observations[i], actions[i], rewards[i], observations_[i], done[i])

    def store_episode(self, episode, steps):
        filename = FILE_PATHS[self.map] + self.action_policy + f"/episode_{episode}.gif"
        
        fig = plt.figure(8, figsize=(steps[0].shape[1] / 64, steps[0].shape[0] / 64), dpi=512)
        fig.suptitle('tick: 0', fontsize=3, fontweight='bold', fontfamily='monospace')
        patch = plt.imshow(steps[0])
        plt.axis('off')

        def animate(i):
            patch.set_data(steps[i])
            fig.suptitle(f'step: {i}', fontsize=3, fontweight='bold')

        gif = animation.FuncAnimation(fig, animate, frames = len(steps), interval=50)
        gif.save(filename, writer='imagemagick', fps=60)

    def run_episode(self, episode, metrics_values):
        metrics = Metrics(self.num_agents, self.visual_radius)
        score = 0
        done = [False]
        observations = self.env.reset()
        losses = []
        step = 0
        steps = []
        # make agent dictionary to store scores for each episode
        for agent in self.agents:    
            agent.scores[episode] = 0
            
        while step < 1000:
            actions = self.choose_actions(observations) 
            new_observations, rewards, done, info = self.env.step(actions)
            metrics.add_step(new_observations, rewards,actions)
            
            # add rewards to each agent
            for i, agent in enumerate(self.agents):
                agent.scores[episode] += rewards[i]
            
            score += sum(rewards)
            losses = self.train_agents(observations, losses, new_observations, actions, rewards, done)

            observations = new_observations
            step += 1
            if episode == 1 or episode % 100 == 0:
                steps.append(self.env.render().copy())
            
        for agent in self.agents:
            agent.decay_epsilon()

        # Save scores
        metrics.calculate_metrics()
        metrics_values.append(metrics)

        with open(self.agent_filename, "a") as f:
            f.write(f"{episode},")
            for agent in self.agents:
                f.write(f"{agent.scores[episode]},")
            f.write("\n")

        if episode == 1 or episode % 100 == 0:
            thread = threading.Thread(target=self.store_episode, args=(episode, steps))
            thread.start()

        print('Episode {} Total Score: {:.2f}, Epsilon {:.2f}'.format(episode, score, self.agents[0].epsilon))
        
        if episode % 5 == 0:
            Metrics.save_as_csv(metrics_values, self.map, self.action_policy)
        