import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math
import sys
import time
import pickle
from tqdm import tqdm


class GA:

    def __init__(self, cityDistances, populationSize, crossoverRate, mutationRate, twoPointCrossover=False):
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
        self.bestFitnesses = []
        self.worstFitnesses = []

        for _ in range(self.populationSize):
            self.currentGen.append(path(len(self.cityDistances)))

    def converge(self, maxGens=1000):
        gens = 0
        start = time.time()
        for _ in tqdm(range(maxGens)):
            gens = gens + 1
            sumFitness = 0
            bestFitIndividual = 0
            worstFitIndividual = sys.maxsize
            for _ in range(self.populationSize // 2):
                daddy = self.binaryTournament()
                mommy = self.binaryTournament()

                babyBoy, babyGirl = self.ox2Crossover(daddy, mommy)

                babyBoy.mutate(self.mutationRate)
                babyGirl.mutate(self.mutationRate)

                self.nextGen = self.nextGen + [babyBoy, babyGirl]
                b1f = babyBoy.fitness(self.cityDistances)
                b2f = babyGirl.fitness(self.cityDistances)
                sumFitness += b1f + b2f
                if b1f > bestFitIndividual:
                    bestFitIndividual = b1f
                if b2f > bestFitIndividual:
                    bestFitIndividual = b2f
                if b1f < worstFitIndividual:
                    worstFitIndividual = b1f
                if b2f < worstFitIndividual:
                    worstFitIndividual = b2f

            self.currentGen = self.nextGen
            self.nextGen = []
            self.generation += 1
            self.generations.append(self.generation)
            self.avgFitnesses.append(sumFitness // self.populationSize)
            self.bestFitnesses.append(bestFitIndividual)
            self.worstFitnesses.append(worstFitIndividual)

    def ox2Crossover(self, male, female):
        if random.randrange(0,100) > self.crossoverRate:
            return male, female
        positions = []

        tenPercent = len(male.genes) * .1
        twentyPercent = len(male.genes) * .2
        fourtyPercent = len(male.genes) * .4
        sixtyPercent = len(male.genes) * .6
        nintyPercent = len(male.genes) * .9

        while len(positions) <= fourtyPercent:
            position = random.randrange(1, len(male.genes))
            if position not in positions:
                positions.append(position)
        positions.sort()

        offspring1 = self.ox2(male.genes, female.genes, positions)
        offspring2 = self.ox2(female.genes, male.genes, positions)

        if len(offspring1.genes) > len(set(offspring1.genes)) or len(offspring2.genes) > len(set(offspring2.genes)):
            print("tour is invalid!")

        return offspring1, offspring2

    def ox2(self, maleGenes, femaleGenes, malePositions):

        offspringGenes = femaleGenes[:]
        femalePositions = []
        crossoverGenes = []
        for position in malePositions:
            maleGene = maleGenes[position]
            crossoverGenes.append(maleGene)
            femalePositions.append(femaleGenes.index(maleGene))

        # femalePositions.sort()
        for i in range(len(femalePositions)):
            offspringGenes[femalePositions[i]] = crossoverGenes[i]

        return path(len(maleGenes), offspringGenes)

    def steveCrossover(self, male, female):
        x = male.genes
        y = female.genes
        cityLen = len(self.cityDistances)
        crossoverPoint = random.randint(cityLen // 2, cityLen - 1)
        crossoverY = y[crossoverPoint:]
        crossoverX = x[crossoverPoint:]
        newX = x[:crossoverPoint] + crossoverY + x[crossoverPoint:]
        newY = y[:crossoverPoint] + crossoverX + y[crossoverPoint:]
        i = 0
        xlen = len(newX)
        ylen = len(newY)
        crossoverPointInList = crossoverPoint
        while i < xlen:
            if i < crossoverPointInList:
                if newX[i] in crossoverY:
                    newX.pop(i)
                    xlen -= 1
                    crossoverPointInList -= 1
                    i -= 1
            elif i >= crossoverPointInList + len(crossoverY):
                if newX[i] in crossoverY:
                    newX.pop(i)
                    xlen -= 1
                    i -= 1
            else:
                i = crossoverPointInList + len(crossoverY) - 1
            i += 1
        i = 0
        crossoverPointInList = crossoverPoint
        while i < ylen:
            if i < crossoverPointInList:
                if newY[i] in crossoverX:
                    newY.pop(i)
                    ylen -= 1
                    crossoverPointInList -= 1
                    i -= 1
            elif i >= crossoverPointInList + len(crossoverX):
                if newY[i] in crossoverX:
                    newY.pop(i)
                    ylen -= 1
                    i -= 1
            else:
                i = crossoverPointInList + len(crossoverX) - 1
            i += 1
        return path(cityLen,newX), path(cityLen,newY)

    def binaryTournament(self):
        candidate1, candidate2 = self.currentGen[random.randrange(0, len(self.currentGen))], \
                                 self.currentGen[random.randrange(0, len(self.currentGen))]
        fit1 = candidate1.fitness(self.cityDistances)
        fit2 = candidate2.fitness(self.cityDistances)

        if fit1 < fit2:
            return candidate1
        else:
            return candidate2

    def displayGraph(self):
        x = np.array(self.generations)
        avgFit = np.array(self.avgFitnesses)
        bestFit = np.array(self.bestFitnesses)
        worstFit = np.array(self.worstFitnesses)
        fig, axes = plt.subplots()
        axes.plot(x, avgFit, color='blue')
        axes.plot(x, bestFit, color='red')
        axes.plot(x, worstFit, color='green')
        axes.set_xlabel('Generations')
        axes.set_ylabel('Fitness')
        fig.suptitle('GA')
        fig.savefig('GA.png')
        # fig.show()
        # input("Press enter to close graph: ")
        plt.close(fig)


class path:

    def __init__(self, numCities, genes=None):
        if not genes:
            self.initGenes(numCities)
        else:
            self.genes = genes

    def initGenes(self, numCities):
        genes = []
        for i in range(1, numCities):
            genes.append(i)
        random.shuffle(genes)
        self.genes = [0] + genes

    def fitness(self, cityDistances):
        sum = 0
        for i in range(len(self.genes)):
            city = self.genes[i]
            nextCity = self.genes[(i + 1) % len(self.genes)]
            sum += cityDistances[city][nextCity]
        return sum

    def mutate(self, rate):
        if random.randrange(0, 100) > rate:
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

def loadPickleFile(inputFile):
    print("Loading data from pickle file")
    start = time.time()
    data = pickle.load(open(inputFile, "rb"))
    print("Data Loaded in {0:.2f}".format(time.time() - start))
    return data


cityDistances = loadPickleFile("wi29.p")
populationSize = 500
crossoverRate = 70
mutationRate = 0

TSPGA = GA(cityDistances, populationSize, crossoverRate, mutationRate)
start = time.time()
TSPGA.converge(1500)
print("Elapsed time: {0:.2f} minutes".format((time.time() - start) / 60))
TSPGA.displayGraph()
