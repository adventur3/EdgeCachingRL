import numpy as np
import random
from zipfRequestsGenerator import RequestGenerator
import time

DEFAULT_RSU_CAPCITY = 1800
DEFAULT_CAR_CAPCITY = 1000

RSU_NUM = 4
#REQUEST_NUM = 10
REQUEST_NUM = 15
REGION_NUM = 8
CAR_NUM = 25


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

        with open('experimentData/RequestSize/15request.txt', 'r') as f:
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

        #self.requestGenerator = RequestGenerator(5, 60, 8, [52, 20, 12, 9, 7])
        #self.requestGenerator = RequestGenerator(5, 60, 8,[35,17,11,9,6,6,5,4,4,3])
        self.requestGenerator = RequestGenerator(5, 60, 8, [31,15,10,8,6,5,4,4,3,3,3,3,2,2,1])
        #self.requestGenerator = RequestGenerator(5, 60, 8, [28,14,10,7,6,5,4,4,3,3,2,2,2,2,2,2,1,1,1,1])
        #self.requestGenerator = RequestGenerator(5, 60, 8, [27,13,9,7,5,4,4,3,3,3,2,2,2,2,2,2,2,1,1,1,1,1,1,1,1])
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

        for i in range(REGION_NUM):
            for j in range(REQUEST_NUM):
                if self.region_request[i][j] == 1:
                    temp_rsu_id = self.region_rsu[i]
                    if self.cache_state[temp_rsu_id][j] == 0 and self.rsu_residual_capcity[temp_rsu_id]>=self.request_size[j]:
                        self.cache_state[temp_rsu_id][j] = 1
                        self.rsu_residual_capcity[temp_rsu_id] -= self.request_size[j]
                    for k in range(len(self.car_location)):
                        if self.car_location[k] == i:
                            if self.car_cache_state[k][j] == 0 and self.car_residual_capcity[k] >= \
                                    self.request_size[j]:
                                self.car_cache_state[k][j] = 1
                                self.car_residual_capcity[k] -= self.request_size[j]

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