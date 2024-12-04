from CommonsGame import *

test = TestModel(num_episodes=1500, map=SmallMap, visual_radius=5, warmup_steps=1000)

test.execute()

