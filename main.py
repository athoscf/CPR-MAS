from CommonsGame import *
from DQN import *

def main():    
    game = TestModel(map=SmallMapWalls, action_policy=ActionPolicies.DEFAULT, num_episodes=5, visual_radius=5, warmup_steps=50)
    game.execute()

if __name__ == "__main__":
    main()