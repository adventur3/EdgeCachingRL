import numpy as np
import random
from zipfRequestsGenerator import RequestGenerator
import time

DEFAULT_RSU_CAPCITY = 2000
DEFAULT_CAR_CAPCITY = 1000

RSU_NUM = 4
REQUEST_NUM = 10
REGION_NUM = 8
CAR_NUM = 20


class CachingEnv:
    def __init__(self):
        self.rsu_num = RSU_NUM
        self.request_num = REQUEST_NUM
        self.region_num = REGION_NUM
        self.car_num = CAR_NUM

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

        with open('experimentData/RegionNeighbor/8regions.txt', 'r') as f:
            temparr = []
            for line in f:
                strlist = str.split(line)
                intlist = list(map(int, strlist))
                temparr.append(intlist)
            self.region_neighbor = np.array(temparr)

        self.requestGenerator = RequestGenerator(5, 60, 8,[35,17,11,9,6,6,5,4,4,3])
        #self.requestFile = open('experimentData/RegionRequest/8region_10request.txt')
        self.cache_state = np.zeros((RSU_NUM, REQUEST_NUM), dtype=int)
        self.car_cache_state = np.zeros((CAR_NUM, REQUEST_NUM), dtype=int)
        self.region_request = self.requestGenerator.generateRegionRequestMatrix()
        self.rsu_capcity = np.ones(RSU_NUM) * DEFAULT_RSU_CAPCITY
        self.car_capcity = np.ones(CAR_NUM) * DEFAULT_CAR_CAPCITY
        self.rsu_residual_capcity = np.ones(RSU_NUM) * DEFAULT_RSU_CAPCITY
        self.car_residual_capcity = np.ones(CAR_NUM) * DEFAULT_CAR_CAPCITY
        self.request_popularity = np.zeros(REQUEST_NUM, dtype=int)
        self.car_location = []
        for i in range(CAR_NUM):
            temp = random.randint(0, REGION_NUM-1)
            self.car_location.append(temp)

        self.n_actions = 5
        self.n_features = RSU_NUM+CAR_NUM+REQUEST_NUM
        self.index_of_core = 2
        self.time1 = 1
        self.time2 = 2
        self.time3 = 3
        self.time4 = 4

    def reset(self):
        self.cache_state = np.zeros((RSU_NUM, REQUEST_NUM), dtype=int)
        self.car_cache_state = np.zeros((CAR_NUM, REQUEST_NUM), dtype=int)
        self.rsu_residual_capcity = np.ones(RSU_NUM) * DEFAULT_RSU_CAPCITY
        self.car_residual_capcity = np.ones(CAR_NUM) * DEFAULT_CAR_CAPCITY
        self.request_popularity = np.zeros(REQUEST_NUM, dtype=int)
        #self.requestFile = open('experimentData/RegionRequest/8region_10request.txt')

        # temparr = []
        # for i in range(self.region_num):
        #     line = self.requestFile.readline()
        #     strlist = str.split(line)
        #     intlist = list(map(int, strlist))
        #     temparr.append(intlist)
        #self.region_request = np.array(temparr)
        self.region_request = self.requestGenerator.generateRegionRequestMatrix()
        self.car_location = []
        for i in range(CAR_NUM):
            temp = random.randint(0, REGION_NUM-1)
            self.car_location.append(temp)

        return np.concatenate([self.rsu_residual_capcity, self.car_residual_capcity, self.request_popularity])

    def step(self, action):
        # temparr = []
        # for i in range(self.region_num):
        #     line = self.requestFile.readline()
        #     strlist = str.split(line)
        #     intlist = list(map(int, strlist))
        #     temparr.append(intlist)
        # self.region_request = np.array(temparr)

        currentRequestCount, hitCacheCount = self.hitCacheCount()

        self.calculatePopularity()
        if action == 0: #global popular content in core rsu
            popularId = np.argmax(self.request_popularity)
            if self.cache_state[self.index_of_core][popularId] == 0 and self.rsu_residual_capcity[self.index_of_core] >= self.request_size[popularId]:
                self.cache_state[self.index_of_core][popularId] = 1
                self.rsu_residual_capcity[self.index_of_core] -= self.request_size[popularId]

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

        elif action == 3: #current popular content in car
            currentContentPopularity = {}
            for i in range(REQUEST_NUM):
                currentContentPopularity[i] = 0
            for i in range(REGION_NUM):
                for j in range(REQUEST_NUM):
                    if self.region_request[i][j] == 1:
                        currentContentPopularity[j] += 1
            popularId = max(currentContentPopularity, key=currentContentPopularity.get)
            tempRequest = self.region_request[:,popularId]
            for i in range(REGION_NUM):
                if tempRequest[i] == 1:
                    for j in range(len(self.car_location)):
                        if self.car_location[j] == i:
                            if self.car_cache_state[j][popularId] == 0 and self.car_residual_capcity[j] >= \
                                    self.request_size[popularId]:
                                self.car_cache_state[j][popularId] = 1
                                self.car_residual_capcity[j] -= self.request_size[popularId]

        else:  #random cache
            tempRSUId = random.randint(0,RSU_NUM-1)
            tempRequestId = random.randint(0,REQUEST_NUM-1)
            if self.cache_state[tempRSUId][tempRequestId] == 0 and self.rsu_residual_capcity[tempRSUId] >= self.request_size[tempRequestId]:
                self.cache_state[tempRSUId][tempRequestId] = 1
                self.rsu_residual_capcity[tempRSUId] -= self.request_size[tempRequestId]
            tempCarId = random.randint(0,CAR_NUM-1)
            tempRequestId = random.randint(0,REQUEST_NUM-1)
            if self.car_cache_state[tempCarId][tempRequestId] == 0 and self.car_residual_capcity[tempCarId] >= self.request_size[tempRequestId]:
                self.car_cache_state[tempCarId][tempRequestId] = 1
                self.car_residual_capcity[tempCarId] -= self.request_size[tempRequestId]

        observation_ = np.concatenate([self.rsu_residual_capcity, self.car_residual_capcity, self.request_popularity])
        store_cost = (np.sum(self.rsu_capcity)+np.sum(self.car_capcity)-np.sum(self.rsu_residual_capcity)-np.sum(self.car_residual_capcity))/(np.sum(self.rsu_capcity)+np.sum(self.car_capcity))
        tranTime_sum = 0
        for i in range(REGION_NUM):
            for j in range(REQUEST_NUM):
                tempRSUId = self.region_rsu[i]
                if self.cache_state[tempRSUId][j] == 1:
                    tranTime_sum += self.time2
                else:
                    tranTime_sum += self.time4
        trans_cost = tranTime_sum/(self.time4*REQUEST_NUM*REGION_NUM)
        cost = store_cost+trans_cost
        #print(store_cost,trans_cost)
        reward = -cost
        self.region_request = self.requestGenerator.generateRegionRequestMatrix()
        self.carmove_step()
        return observation_, reward, currentRequestCount, hitCacheCount

    def carmove_step(self):
        for i in range(len(self.car_location)):
            current_location = self.car_location[i]
            tempint = random.randint(1, 100)
            if tempint > 90:
                candi_location = []
                for j in range(len(self.region_neighbor[current_location])):
                    if current_location != j and self.region_neighbor[current_location][j] == 1:
                        candi_location.append(j)
                candi_location_choose = random.randint(0, len(candi_location)-1)
                self.car_location[i] = candi_location[candi_location_choose]



    def hitCacheCount(self):
        requestCount = 0
        cacheHitCount = 0
        for i in range(len(self.region_request)):
            for j in range(len(self.region_request[0])):
                if self.region_request[i][j] == 1:
                    requestCount += 1
                    correspondRsuId = self.region_rsu[i]
                    if self.cache_state[correspondRsuId][j] ==1:
                        cacheHitCount += 1
                    else:
                        for car_id in range(len(self.car_location)):
                            if self.car_location[car_id] == i and self.car_cache_state[car_id][j] == 1:
                                cacheHitCount += 1
                                break

        return requestCount, cacheHitCount

    def calculatePopularity(self):
        for i in range(REGION_NUM):
            for j in range(REQUEST_NUM):
                if self.region_request[i][j] == 1:
                    self.request_popularity[j] += 1

if __name__ == '__main__':
    c = CachingEnv()
    #a = c.reset()
    #print(a)