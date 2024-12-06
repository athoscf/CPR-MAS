from CommonsGame import *
from DQN import *

def main():    
    game = TestModel(map=OpenMapV2, action_policy=ActionPolicies.MIXED, num_episodes=1201, visual_radius=5, warmup_steps=5000)
    game.execute()

if __name__ == "__main__":
    main()