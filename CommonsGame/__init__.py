from gym.envs.registration import register
from CommonsGame import *
from CommonsGame.envs import *
from CommonsGame.resources import *
from CommonsGame.objects import *
from CommonsGame.model import *
import gym
import numpy as np

register(
    id='CommonsGame-v0',
    entry_point='CommonsGame.envs:CommonsGame',
)