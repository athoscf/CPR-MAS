from CommonsGame import *
from DQN import *

def main():    
    # Original maps
    original_small_map = TestModel(num_episodes=3, map=SmallMap, visual_radius=5, warmup_steps=1000)
    original_small_map.execute()
    
    original_open_map = TestModel(num_episodes=15, map=OpenMap, visual_radius=5, warmup_steps=1000)
    original_open_map.execute()
    
    # New added maps with walls
    single_entrance_map = TestModel(num_episodes=15, map=SingleEntranceMap, visual_radius=5, warmup_steps=1000)
    single_entrance_map.execute()
    
    unequal_entrance_map = TestModel(num_episodes=15, map=UnequalEntranceMap, visual_radius=5, warmup_steps=1000)
    unequal_entrance_map.execute()
    
    multiple_entrance_map = TestModel(num_episodes=15, map=MultipleEntranceMap, visual_radius=5, warmup_steps=1000)
    multiple_entrance_map.execute()
    
    # Agents with gifting enabled
    gifting_open_map = TestModel(num_episodes=15, map=OpenMap, visual_radius=5, warmup_steps=1000, action_policy=ActionPolicies.GIFT_ONLY)
    gifting_open_map.execute()

    # Agents with different policies and actions
    agents_with_different_policies = TestModel(num_episodes=15, map=OpenMap, visual_radius=5, warmup_steps=1000, action_policy=ActionPolicies.MIXED)
    agents_with_different_policies.execute()

if __name__ == "__main__":
    main()