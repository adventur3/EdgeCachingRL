import numpy as np

REGION_NUM = 8
REQUEST_NUM = 10

if __name__ == "__main__":
    f = open("experimentData/RegionRequest/8region_10request.txt", "w")
    for round in range(300000):
        matrix = -1 + 2 * np.random.random((REGION_NUM,REQUEST_NUM))
        for i in range(REGION_NUM):
            for j in range(REQUEST_NUM):
                if matrix[i][j] > 0:
                    if j==REQUEST_NUM-1:
                        f.write("1")
                    else:
                        f.write("1 ")
                else:
                    if j==REQUEST_NUM-1:
                        f.write("0")
                    else:
                        f.write("0 ")
            f.write("\n")