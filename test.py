from CommonsGame import *

test = TestModel(num_episodes=5000, map=SmallMap, visual_radius=5, warmup_steps=1000)

test.execute()

