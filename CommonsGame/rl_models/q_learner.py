import numpy as np
import random

class QLearner:
    def __init__(self, id, num_states, num_actions, learning_rate=0.1, discount_factor=0.99, exploration_rate=1.0, exploration_decay=0.99):
        """
        Initialize the Q-learning agent.
        :param id: Unique identifier for the agent.
        :param num_states: Number of states in the environment.
        :param num_actions: Number of possible actions.
        :param learning_rate: Step size for updating Q-values (alpha).
        :param discount_factor: Discount factor for future rewards (gamma).
        :param exploration_rate: Initial exploration rate for epsilon-greedy policy (epsilon).
        :param exploration_decay: Factor to decay epsilon after each step.
        """
        self.id = id
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.q_table = np.zeros((num_states, num_actions))  # Q-values initialized to zero
        
    def _state_to_index(self, state):
        """Convert observation (state) to a hashable index."""
        if state is None:
            return -1  # Handle None state
        return hash(state) % 1000000  # Simplified; modify if needed for large state spaces
    
    def select_action(self, state):
        """
        Select an action using an epsilon-greedy policy.
        :param state: Current state of the agent.
        :return: Selected action.
        """
        if random.random() < self.epsilon:
            # Explore: Choose a random action
            return random.randint(0, self.num_actions - 1)
        else:
            # Exploit: Choose the best-known action
            return np.argmax(self.q_table[self._state_to_index(state)])
    
    def update_q_value(self, state, action, reward, next_state, done):
        """
        Update the Q-value based on the observed transition.
        :param state: Current state.
        :param action: Action taken.
        :param reward: Reward received.
        :param next_state: Next state after taking the action.
        :param done: Whether the episode has ended.
        """
        state_index = self._state_to_index(state)
        next_state_index = self._state_to_index(next_state)
        best_next_action = np.argmax(self.q_table[next_state_index]) if not done else 0
        td_target = reward + self.gamma * self.q_table[next_state_index, best_next_action] * (1 - done)
        td_error = td_target - self.q_table[state_index, action]
        self.q_table[state_index, action] += self.alpha * td_error
        # Decay exploration rate at the end of each episode
        self._decay_exploration()
    
    def _decay_exploration(self):
        """
        Decay the exploration rate (epsilon).
        """
        self.epsilon *= self.epsilon_decay
        self.epsilon = max(self.epsilon, 0.01)  # Ensures epsilon doesn't fall below a minimum threshold

    

# Example usage
if __name__ == "__main__":
    num_states = 10
    num_actions = 4
    agent = QLearner(num_states, num_actions)

    # Simulate a simple environment
    for episode in range(10):  # Example: 10 episodes
        state = random.randint(0, num_states - 1)
        done = False
        while not done:
            action = agent.select_action(state)
            next_state = random.randint(0, num_states - 1)
            reward = random.random()
            done = random.random() < 0.1  # 10% chance of ending episode
            agent.update_q_value(state, action, reward, next_state, done)
            state = next_state

        agent.decay_exploration()
        print(f"Episode {episode + 1}: Exploration rate: {agent.epsilon}")
