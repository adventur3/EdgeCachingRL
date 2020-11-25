import numpy as np
import time

RSU_NUM = 4
REQUEST_NUM = 4
REGION_NUM = 40

class CachingEnv:
    def __init__(self):
        self.action_space = ['polular_local', 'current_local', 'current_core', 'polular_core', 'none']
        self.n_actions = len(self.action_space)
        self.n_features = 2
        self.cache_state = np.zeros((RSU_NUM, REQUEST_NUM))
        self.rsunnect = np.zeros((RSU_NUM, RSU_NUM))
        self.rsu_region = np.zeros((RSU_NUM, REGION_NUM))
        self.region_request = np.zeros((REGION_NUM, REQUEST_NUM))
        self.request_popularity = np.zeros(REQUEST_NUM)
        self.index_of_core = 2

    def step(self, action):
        if action == 'popular_local':

        elif action == 'current_local':

        elif action == 'current_core':

        elif action == 'popular_core':

        elif action == 'none':




if __name__ == '__main__':
    cacheEnv = CachingEnv()
    print(cacheEnv.cache_state)
    print(cacheEnv.cache_state[1][3])
    print(cacheEnv.request_popularity)