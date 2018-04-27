# Start of code for processing TSP data for tours of countries
# http://www.math.uwaterloo.ca/tsp/world/countries.html
import math
import sys
import pickle
import os
from tqdm import tqdm


def processTour(fileName, numPreLines):
    infile = open(fileName, "r")
    dataD = {}
    listData = []
    for i in range(numPreLines):
        infile.readline()
    for line in infile:
        lstVals = line.split()
        if len(lstVals) > 1:
            dataD[int(lstVals[0])] = [float(lstVals[1]), float(lstVals[2])]
            listData.append([float(lstVals[1]), float(lstVals[2])])


    TSPMatrix = []
    for row in tqdm(range(len(listData))):
        r = []
        for col in range(len(listData)):
            d = round(
                math.sqrt((listData[row][0] - listData[col][0]) ** 2 + (listData[row][1] - listData[col][1]) ** 2))
            r.append(d)
        TSPMatrix.append(r)
    return sys.getsizeof(dataD), sys.getsizeof(listData), TSPMatrix


szD, szL, tsp = processTour("sw24978.tsp", 7)
# pickle.dump(tsp, open("sw24978-INTS.p", "wb"))
# print(os.path.getsize("sw24978-INTS.p"))
# print(test)


# Code for outputting to text file.
# outFile = open("matrix.txt", "w")
# for row in range(len(tsp)):
#     # outFile.write(str(tsp[row]).rjust(300)+"\n")
#     outFile.write(str(tsp[row]) + "\n")
# outFile.close()
