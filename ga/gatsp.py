import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import sys
import time



def processTour(fileName, numPreLines):
    infile = open(fileName, "r")
    dataD = {}
    listData = []
    for i in range(numPreLines):
        infile.readline()
    for line in infile:
        lstVals = line.split()
        if len(lstVals) > 1:
            # dataD[int(lstVals[0])]=[int(float(lstVals[1])*10000),int(float(lstVals[2])*10000)]
            # listData.append([int(float(lstVals[1])*10000),int(float(lstVals[2])*10000)])
            dataD[int(lstVals[0])] = [float(lstVals[1]), float(lstVals[2])]
            listData.append([float(lstVals[1]), float(lstVals[2])])

    # dataD=[]
    # listData=[[10,10],[8,8],[5,5],[6,6],[12,12],[2,2]]
    print(listData[:1])
    start = time.time()
    TSPMatrix = []
    for row in range(len(listData)):
        r = []
        # for col in range(row,len(listData)):
        for col in range(len(listData)):
            # print((listData[row][0]-listData[col][0])**2)
            d = round(
                (math.sqrt((listData[row][0] - listData[col][0]) ** 2 + (listData[row][0] - listData[col][0]) ** 2)), 2)
            print(d)
            r.append(d)
        TSPMatrix.append(r)
        if row % 1000 == 0:
            print(row, time.time() - start)
    return sys.getsizeof(dataD), sys.getsizeof(listData), TSPMatrix


class GA:

    def __init__(self,  cityDistances, populationSize, crossoverRate, mutationRate, twoPointCrossover=False):
        self.cityDistances = cityDistances

        self.mutationRate = mutationRate
        self.crossoverRate = crossoverRate
        self.twoPointCrossover = twoPointCrossover
        self.populationSize = populationSize

        self.currentGen = []
        self.nextGen = []
        self.generation = 0

        # graph stuff
        self.generations = []
        self.avgFitnesses = []

        for _ in range(self.populationSize):
            self.currentGen.append(path(len(self.cityDistances)))

    def converge(self, maxGens=1000):
        while self.generation < maxGens:
            sumFitness = 0
            for _ in range(self.populationSize // 2):
                male = self.binaryTournament()
                female = self.binaryTournament()

                baby1, baby2 = self.crossover(male, female)

                baby1.mutate(self.mutationRate)
                baby2.mutate(self.mutationRate)
                # print(baby1.genes)
                # print(baby2.genes)

                self.nextGen = self.nextGen + [baby1, baby2]
                sumFitness += baby1.fitness(self.cityDistances) + baby2.fitness(self.cityDistances)

            self.currentGen = self.nextGen
            self.nextGen = []
            self.generation += 1
            self.generations.append(self.generation)
            self.avgFitnesses.append(sumFitness // self.populationSize)

    def crossover(self, male, female):
        if self.crossoverRate < random.randrange(0, 100):
            return male, female

        idx = random.randrange(1,len(male.genes))

        maleBaby = [0] + ([-1] * (len(male.genes)-1))
        femaleBaby = [0] + ([-1] * (len(male.genes)-1))

        for i in range(idx, len(male.genes)):
            maleBaby[i] = female.genes[i]
            femaleBaby[i] = male.genes[i]

        for i in range(1, idx):
            mGene = male.genes[i]
            while mGene in maleBaby:
                mappedGene = male.genes[female.genes.index(male.genes[i])]
                if mGene == mappedGene:
                    mGene = random.randrange(1, len(male.genes))
                else:
                    mGene = mappedGene
            maleBaby[i] = mGene

            fGene = female.genes[i]
            while fGene in femaleBaby:
                mappedGene = female.genes[male.genes.index(female.genes[i])]
                if fGene == mappedGene:
                    fGene = random.randrange(1, len(male.genes))
                else:
                    fGene = mappedGene
            femaleBaby[i] = fGene

        if len(femaleBaby) != len(set(femaleBaby)) or len(maleBaby) != len(set(maleBaby)):
            print("There was a duplicate value")

        if maleBaby[0] != 0 or femaleBaby[0] != 0:
            print("First city is not Houston")

        return path(len(self.cityDistances), maleBaby), path(len(self.cityDistances), femaleBaby)

    def binaryTournament(self):
        candidate1, candidate2 = self.currentGen[random.randrange(0, len(self.currentGen))], \
                                 self.currentGen[random.randrange(0, len(self.currentGen))]
        if candidate1.fitness(self.cityDistances) < candidate2.fitness(self.cityDistances):
            return candidate1
        else:
            return candidate2

    def displayGraph(self):
        x = np.array(self.generations)
        y = np.array(self.avgFitnesses)
        fig, axes = plt.subplots()
        axes.plot(x, y, color='blue')
        axes.set_xlabel('Generations')
        axes.set_ylabel('Fitness')
        fig.suptitle('GA')
        fig.savefig('GA.png')
        fig.show()
        input("Press enter to close graph: ")
        plt.close(fig)


class path:

    def __init__(self, numCities, genes=None):
        if not genes:
            self.initGenes(numCities)
        else:
            self.genes = genes

    def initGenes(self, numCities):
        genes = []
        for i in range(1,numCities):
            genes.append(i)
        random.shuffle(genes)
        self.genes = [0] + genes


    def fitness(self, cityDistances):
        sum = 0
        for i in range(len(self.genes)):
            city = self.genes[i]
            nextCity = self.genes[(i + 1) % len(self.genes)]
            sum += cityDistances[city][nextCity]
        # print(1/sum*100000)
        return sum

    def mutate(self, rate):
        if rate < random.randrange(0, 100):
            pass

        chromosome1 = random.randrange(1, len(self.genes))
        chromosome2 = random.randrange(1, len(self.genes))

        tmp = self.genes[chromosome1]
        self.genes[chromosome1] = self.genes[chromosome2]
        self.genes[chromosome2] = tmp


# cityNames = ["Houston", "Dallas", "Austin", "Abilene", "Waco"]
# cityDistances = [[0, 241, 162, 351, 183],
#                  [241, 0, 202, 186, 97],
#                  [162, 202, 0, 216, 106],
#                  [351, 186, 216, 0, 186],
#                  [183, 97, 106, 186, 0]]

szD, szL, cityDistances = processTour("wi29.tsp", 7)


test = GA(cityDistances, 50, 90, 1)
test.converge(1000)
test.displayGraph()