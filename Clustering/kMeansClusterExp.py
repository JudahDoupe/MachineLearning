import random
import math
import sys


def euclidD(point1, point2):
    sum = 0
    for index in range(len(point1)):
        diff = (point1[index] - point2[index]) ** 2
        sum = sum + diff
    euclidDistance = math.sqrt(sum)
    return euclidDistance


def createCentroids(k, datadict):
    centroids = []
    centroidCount = 0
    centroidKeys = []
    while centroidCount < k:
        rkey = random.randint(1, len(datadict))
        if rkey not in centroidKeys:
            centroids.append(datadict[rkey])
            centroidKeys.append(rkey)
            centroidCount = centroidCount + 1
    return centroids


def createClusters(k, centroids, datadict, repeats, debugFlag, graphFlag):
    for apass in range(repeats):
        if debugFlag:
            print("***************PASS", apass, "***************************")
        clusters = []
        for i in range(k):
            clusters.append([])
        if debugFlag:
            print("Initial Clusters = ", clusters)
            print("Data dictionary = ", datadict)
            if debugFlag:
                print()
            print("Centroids = ", centroids)
        for akey in datadict:  # per data point
            distances = []
            for clusterIndex in range(k):  # per centroid
                dist = euclidD(datadict[akey], centroids[clusterIndex])
                distances.append(dist)
            if debugFlag:
                print("Distances for point ", akey, " = [", end="")
                outString = ""
                for z in range(len(distances)):
                    outString += ("%3.1f, " % distances[z])
                outString = outString[:-2] + "]"
                print(outString)
            mindist = min(distances)
            index = distances.index(mindist)
            clusters[index].append(akey)
            displayClusters = []
            for cluster in clusters:
                clust = []
                for key in cluster:
                    clust.append(datadict[key])
                displayClusters.append(clust)
        if debugFlag:
            print("Clusters (keys) are now = ", clusters)
            # print("Clusters (values) are now = ",displayClusters)
            for i in range(k):
                print("Cluster Display ", i, " = ", displayClusters[i])
            junk = input("press enter to continue . . .")
        dimensions = len(datadict[1])
        for clusterIndex in range(k):
            sums = [0] * dimensions
            for akey in clusters[clusterIndex]:
                datapoints = datadict[akey]
                for ind in range(len(datapoints)):
                    sums[ind] = sums[ind] + datapoints[ind]
            if debugFlag:
                print("Summed distances for cluster ", clusterIndex, " is = ", sums)
            for ind in range(len(sums)):
                clusterLen = len(clusters[clusterIndex])
                if clusterLen != 0:
                    sums[ind] = sums[ind] / clusterLen
            centroids[clusterIndex] = sums
        if debugFlag:
            print("The new centroids are ", centroids)
        count = 1
        for c in clusters:
            if debugFlag:
                print("CLUSTER ", count)
            for key in c:
                if debugFlag:
                    print(datadict[key], end="  ")
            if debugFlag:
                print()
            count = count + 1
    return clusters


def getMinsMaxes(datadict):
    xmin = sys.maxint
    xmax = sys.minint
    ymin = sys.maxint
    ymax = sys.minint
    for key in datadict:
        if datadict[key][0] < xmin:
            xmin = datadict[key][0]
        if datadict[key][0] > xmax:
            xmin = datadict[key][0]


def clusterFI(k, numPasses):
    dataDict, classDict = readFIFile("IrisNormedLong.txt")
    centroids = createCentroids(k, dataDict)
    clusters = createClusters(k, centroids, dataDict, numPasses, False, False)
    # print(clusters)
    return clusters, classDict


def readFIFile(filename):
    datafile = open(filename, "r")
    datadict = {}
    classificationdict = {}
    key = 0
    for aline in datafile:
        items = aline.split()
        key = key + 1
        f1 = float(items[0])
        f2 = float(items[1])
        f3 = float(items[2])
        f4 = float(items[3])
        classification = float(items[4])
        datadict[key] = [f1, f2, f3, f4]
        classificationdict[key] = [classification]
    return datadict, classificationdict

def accuracy(clusters, classDict):
    classificationFrequencies = []
    i = 0
    for cluster in clusters:
        classificationFrequencies.append({})
        for j in range(len(clusters)):
            classificationFrequencies[i][j] = 0
        for id in cluster:
            classification = int(classDict[id][0])
            classificationFrequencies[i][classification] += 1
        i = i + 1



    classificationClusters = {}
    for i in range(len(clusters)):
        maxVal = 0
        bestClassification = {}
        for classification in classificationFrequencies:
            if classification[i] >= maxVal:
                maxVal = classification[i]
                bestClassification = classification
        classificationClusters[i] = bestClassification.copy()
        classificationFrequencies.remove(bestClassification)

    print(classificationClusters)
    for classification, cluster in classificationClusters.items():
        total = 0

        for i, val in cluster.items():
            total += val


        print("Cluster {0} Accuracy: {1:.2f}%".format(classification,  100 *(cluster[classification]/ total)) )



clusters, classDict = clusterFI(3, 10)
accuracy(clusters,classDict)


