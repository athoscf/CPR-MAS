from gym.envs.registration import register
from CommonsGame import *
from CommonsGame.envs import *
from CommonsGame.resources import *
from CommonsGame.objects import *

register(
    id='CommonsGame-v0',
    entry_point='CommonsGame.envs:CommonsGame',
)