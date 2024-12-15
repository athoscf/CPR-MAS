from CommonsGame import *
from DQN import *

def main():    
    game = TestModel(map=SmallMap, action_policy=ActionPolicies.TAG_AND_GIFT, num_episodes=1501, visual_radius=5, warmup_steps=5000)
    game.execute()

if __name__ == "__main__":
    main()