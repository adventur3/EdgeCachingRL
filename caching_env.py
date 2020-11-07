import numpy as np
import time

RSU_COUNT = 40
REQUEST_TYPE_COUNT = 4
REQUEST_NUM = 4

class CachingEnv:
    def __init__(self):
        self.action_space = ['u', 'd', 'l', 'r']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.title('maze')
        self._build_maze()

    def step(self, action):
        #do action
        #return s_, reward, done