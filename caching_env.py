import numpy as np
import time

RSU_NUM = 4
REQUEST_NUM = 4
REGION_NUM = 40
T_RSU = 1
T_CLOUD = 2

class CachingEnv:
    def __init__(self):
        self.cache_state = np.zeros((RSU_NUM, REQUEST_NUM))
        self.region_request = np.zeros((REGION_NUM, REQUEST_NUM))
        self.rsu_connect = np.zeros((RSU_NUM, RSU_NUM))

        self.rsu_capcity = np.zeros(RSU_NUM)
        self.rsu_residual_capcity = np.zeros(RSU_NUM)
        self.request_popularity = np.zeros(REQUEST_NUM)
        self.request_size = np.zeros(REQUEST_NUM)
        self.index_of_core = 2

    def reset(self):
        self.cache_state = np.zeros((RSU_NUM, REQUEST_NUM))
        self.region_request = np.zeros((REGION_NUM, REQUEST_NUM))
        self.rsu_connect = np.zeros((RSU_NUM, RSU_NUM))

        self.rsu_capcity = np.zeros(RSU_NUM)
        self.rsu_residual_capcity = np.zeros(RSU_NUM)
        self.request_popularity = np.zeros(REQUEST_NUM)
        self.index_of_core = 2

    def step(self, action):
        _s = action
        trans_efficiency_sum = 0
        capcity_efficiency_sum = 0
        capcity_sum = 0
        for i in range(RSU_NUM):
            capcity_sum += self.rsu_capcity[i]
            for j in range(REQUEST_NUM):
                t = T_RSU/(_s[i][j]*T_RSU+(1-_s[i][j])*T_CLOUD)
                trans_efficiency_sum += t
                capcity_efficiency_sum += _s[i][j]*self.request_size[j]
        ucche = trans_efficiency_sum/(capcity_efficiency_sum/capcity_sum)
        return _s,ucche




if __name__== '__main__':
    cacheEnv = CachingEnv()
    print(cacheEnv.cache_state)
    print(cacheEnv.cache_state[1][3])
    print(cacheEnv.request_popularity)