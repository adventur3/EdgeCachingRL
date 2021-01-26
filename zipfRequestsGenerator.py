import numpy as np
import random

rank1_count = 1200
num_count = 10
arr = np.zeros(num_count)
for i in range(num_count):
    arr[i] = rank1_count/(i+1)
print(arr,sum(arr))

class RequestGenerator:
    def __init__(self, rank1_count, minNum_each_slot, maxNum_each_slot, probability_arr):
        self.rank1_count = rank1_count
        self.minNum_each_slot = minNum_each_slot
        self.maxNum_each_slot = maxNum_each_slot
        self.probability_arr = probability_arr

    def generateSlotRequests(self):
        num = random.randint(self.minNum_each_slot, self.maxNum_each_slot)
        request_arr = np.zeros(num)
        for i in range(num):
            request_arr[i] = self.random_index(self.probability_arr)
        return request_arr

    def random_index(self, probability_arr):
        start = 0
        index = 0
        randnum = random.randint(1, sum(probability_arr))
        for index, scope in enumerate(probability_arr):
            start += scope
            if randnum <= start:
                break
        return index

if __name__ == '__main__':
    requestGen = RequestGenerator(1200, 5, 60, [35,17,11,9,6,6,5,4,4,3])
    print(requestGen.generateSlotRequests())