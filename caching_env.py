import numpy as np
import random
import time

DEFAULT_RSU_CAPCITY = 1000

RSU_NUM = 4
REQUEST_NUM = 10
REGION_NUM = 8


class CachingEnv:
    def __init__(self):
        self.rsu_num = RSU_NUM
        self.request_num = REQUEST_NUM
        self.region_num = REGION_NUM

        with open('experimentData/RegionRsu/8region_4rsu.txt', 'r') as f:
            line = f.readline()
            strlist = str.split(line)
            intlist = list(map(int, strlist))
            self.region_rsu = np.array(intlist)

        with open('experimentData/RsuConnect/4rsuConnect.txt', 'r') as f:
            temparr = []
            for line in f:
                strlist = str.split(line)
                intlist = list(map(int, strlist))
                temparr.append(intlist)
            self.rsu_connect = np.array(temparr)

        with open('experimentData/RequestSize/10request.txt', 'r') as f:
            line = f.readline()
            strlist = str.split(line)
            intlist = list(map(int, strlist))
            self.request_size = np.array(intlist)

        self.requestFile = open('experimentData/RegionRequest/8region_10request.txt')
        self.cache_state = np.zeros((RSU_NUM, REQUEST_NUM), dtype=int)
        self.region_request = np.zeros((REGION_NUM, REQUEST_NUM), dtype=int)
        self.rsu_capcity = np.ones(RSU_NUM) * DEFAULT_RSU_CAPCITY
        self.rsu_residual_capcity = np.ones(RSU_NUM) * DEFAULT_RSU_CAPCITY
        self.request_popularity = np.zeros(REQUEST_NUM, dtype=int)

        self.n_actions = 4
        self.n_features = RSU_NUM+REQUEST_NUM
        self.index_of_core = 2
        self.time1 = 1
        self.time2 = 2
        self.time3 = 3
        self.time4 = 4

    def reset(self):
        self.cache_state = np.zeros((RSU_NUM, REQUEST_NUM), dtype=int)
        self.rsu_residual_capcity = np.ones(RSU_NUM) * DEFAULT_RSU_CAPCITY
        self.request_popularity = np.zeros(REQUEST_NUM, dtype=int)
        self.requestFile = open('experimentData/RegionRequest/8region_10request.txt')

        temparr = []
        for i in range(self.region_num):
            line = self.requestFile.readline()
            strlist = str.split(line)
            intlist = list(map(int, strlist))
            temparr.append(intlist)
        self.region_request = np.array(temparr)


        return np.concatenate([self.rsu_residual_capcity, self.request_popularity])

    def step(self, action):
        temparr = []
        for i in range(self.region_num):
            line = self.requestFile.readline()
            strlist = str.split(line)
            intlist = list(map(int, strlist))
            temparr.append(intlist)
        self.region_request = np.array(temparr)

        self.calculatePopularity()
        if action == 0: #global popular content in core rsu
            popularId = np.argmax(self.request_popularity)
            if self.cache_state[self.index_of_core][popularId] == 0 and self.rsu_residual_capcity[self.index_of_core] >= self.request_size[popularId]:
                self.cache_state[self.index_of_core][popularId] = 1
                self.rsu_residual_capcity -= self.request_size[popularId]

        elif action == 1: #current popular content in local rsu
            currentContentPopularity = {}
            for i in range(REQUEST_NUM):
                currentContentPopularity[i] = 0
            for i in range(REGION_NUM):
                for j in range(REQUEST_NUM):
                    if self.region_request[i][j] == 1:
                        currentContentPopularity[j] += 1
            popularId = max(currentContentPopularity,key=currentContentPopularity.get)
            tempRequest = self.region_request[:,popularId]
            for i in range(REGION_NUM):
                if tempRequest[i] == 1:
                    tempRSUId = self.region_rsu[i]
                    if self.cache_state[tempRSUId][popularId] == 0 and self.rsu_residual_capcity[tempRSUId]>=self.request_size[popularId]:
                        self.cache_state[tempRSUId][popularId] = 1
                        self.rsu_residual_capcity[tempRSUId] -= self.request_size[popularId]

        elif action == 2: #current popular content in core rsu
            currentContentPopularity = {}
            for i in range(REQUEST_NUM):
                currentContentPopularity[i] = 0
            for i in range(REGION_NUM):
                for j in range(REQUEST_NUM):
                    if self.region_request[i][j] == 1:
                        currentContentPopularity[j] += 1
            popularId = max(currentContentPopularity, key=currentContentPopularity.get)
            if self.cache_state[self.index_of_core][popularId] == 0 and self.rsu_residual_capcity[self.index_of_core] >= self.request_size[popularId]:
                self.cache_state[self.index_of_core][popularId] = 1
                self.rsu_residual_capcity[self.index_of_core] -= self.request_size[popularId]

        else:  #random cache
            tempRSUId = random.randint(0,RSU_NUM-1)
            tempRequestId = random.randint(0,REQUEST_NUM-1)
            if self.cache_state[tempRSUId][tempRequestId] == 0 and self.rsu_residual_capcity[tempRSUId] >= self.request_size[tempRequestId]:
                self.cache_state[tempRSUId][tempRequestId] = 1
                self.rsu_residual_capcity[tempRSUId] -= self.request_size[tempRequestId]

        observation_ = np.concatenate([self.rsu_residual_capcity, self.request_popularity])
        store_cost = (np.sum(self.rsu_capcity)-np.sum(self.rsu_residual_capcity))/np.sum(self.rsu_capcity)
        tranTime_sum = 0
        for i in range(REGION_NUM):
            for j in range(REQUEST_NUM):
                tempRSUId = self.region_rsu[i]
                if self.cache_state[tempRSUId][j] == 1:
                    tranTime_sum += self.time2
                else:
                    tranTime_sum += self.time4
        trans_cost = tranTime_sum/self.time4*RSU_NUM
        cost = store_cost+trans_cost
        reward = -cost

        return observation_, reward

    def calculatePopularity(self):
        for i in range(REGION_NUM):
            for j in range(REQUEST_NUM):
                if self.region_request[i][j] == 1:
                    self.request_popularity[j] += 1
