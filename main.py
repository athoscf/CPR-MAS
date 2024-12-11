from CommonsGame import *
from DQN import *

def main():    
    game = TestModel(map=SmallMapWalls, action_policy=ActionPolicies.TAG_AND_GIFT, num_episodes=2001, visual_radius=5, warmup_steps=5000)
    game.execute()

    game = TestModel(map=SmallMapWalls, action_policy=ActionPolicies.MIXED, num_episodes=2001, visual_radius=5, warmup_steps=5000)
    game.execute()

if __name__ == "__main__":
    main()