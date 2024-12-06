from CommonsGame import *
from DQN import *

def main():    
    #original_open_map = TestModel(num_episodes=15, map=OpenMap, visual_radius=5, warmup_steps=1000)
    #original_open_map.execute()
    # Original maps
    # original_small_map = TestModel(num_episodes=15, map=SmallMap, visual_radius=5, warmup_steps=1000, csv_filename="small_map_result.csv")
    # original_small_map.execute()
    
    game = TestModel(map=OpenMapV2, action_policy=ActionPolicies.MIXED, num_episodes=5, visual_radius=5, warmup_steps=5000)
    game.execute()
    
    # # New added maps with walls
    # single_entrance_map = TestModel(num_episodes=15, map=SingleEntranceMap, visual_radius=5, warmup_steps=1000, csv_filename="single_entrance_map_result.csv")
    # single_entrance_map.execute()
    
    # unequal_entrance_map = TestModel(num_episodes=15, map=UnequalEntranceMap, visual_radius=5, warmup_steps=1000, csv_filename="unequal_entrance_map_result.csv")
    # unequal_entrance_map.execute()
    
    # multiple_entrance_map = TestModel(num_episodes=1500, map=MultipleEntranceMap, visual_radius=5, warmup_steps=50000)
    # multiple_entrance_map.execute()
    
    # # Agents with gifting enabled
    # gifting_open_map = TestModel(num_episodes=15, map=OpenMap, visual_radius=5, warmup_steps=1000, action_policy=ActionPolicies.GIFT_ONLY, csv_filename="gifting_open_map_result.csv")
    # gifting_open_map.execute()

    # # Agents with different policies and actions
    # agents_with_different_policies = TestModel(num_episodes=15, map=OpenMap, visual_radius=5, warmup_steps=1000, action_policy=ActionPolicies.MIXED, csv_filename="mixed_policies_result.csv")
    # agents_with_different_policies.execute()

if __name__ == "__main__":
    main()