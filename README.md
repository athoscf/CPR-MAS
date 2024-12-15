This is an OpenAI gym implementation of the Commons Game, a multi-agent environment proposed in [A multi-agent reinforcement learning model of common-pool resource appropriation](https://arxiv.org/abs/1707.06600) using [pycolab](https://github.com/deepmind/pycolab) as game engine.

We will use a Deep Q-Network (DQN) to train the agents in this environment, allowing them to learn optimal strategies for resource appropriation while balancing cooperation and competition. The DQN algorithm leverages deep learning to approximate the Q-value function, enabling agents to make decisions in this multi-agent setting.

## Installation

To install `cd` to the directory of the repository and run `pip install -e .`

## Usage

The file `main.py` contains a simple usage example where you can modify the map config, the size of its field of vision, the number of episodes and the action policy. To run the file `cd` to the directory of the repository and run `python main.py`.

