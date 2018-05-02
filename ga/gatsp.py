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

        for _ in range(self.populationSize):
            self.currentGen.append(path(len(self.cityDistances)))

    def converge(self, maxGens=1000):
        gens = 0
        start = time.time()
        for _ in tqdm(range(maxGens)):
            gens = gens + 1
            sumFitness = 0
            for _ in range(self.populationSize // 2):
                male = self.binaryTournament()
                female = self.binaryTournament()

                baby1, baby2 = self.crossover(male, female)

                baby1.mutate(self.mutationRate)
                baby2.mutate(self.mutationRate)

                self.nextGen = self.nextGen + [baby1, baby2]
                sumFitness += baby1.fitness(self.cityDistances) + baby2.fitness(self.cityDistances)

            self.currentGen = self.nextGen
            self.nextGen = []
            self.generation += 1
            self.generations.append(self.generation)
            self.avgFitnesses.append(sumFitness // self.populationSize)


    def crossover(self, male, female):
        positions = []

        twentyPercent = len(male.genes) * .2
        fourtyPercent = len(male.genes) * .4
        sixtyPercent = len(male.genes) * .6

        while len(positions) <= fourtyPercent:
            position = random.randrange(1, len(male.genes))
            if position not in positions:
                positions.append(position)
        positions.sort()

        # positions = [1, 2, 5]

        ofspring1 = self.ox2(male.genes, female.genes, positions)
        ofspring2 = self.ox2(female.genes, male.genes, positions)

        return ofspring1, ofspring2

    def ox2(self, maleGenes, femaleGenes, malePositions):

        offspingGenes = femaleGenes[:]
        femalePositions = []
        crossoverGenes = []
        for position in malePositions:
            maleGene = maleGenes[position]
            crossoverGenes.append(maleGene)
            femalePositions.append(femaleGenes.index(maleGene))

        femalePositions.sort()
        for i in range(len(femalePositions)):
            offspingGenes[femalePositions[i]] = crossoverGenes[i]

        return path(len(maleGenes), offspingGenes)

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

def loadPickleFile(inputFile):
    print("Loading data from pickle file")
    start = time.time()
    data = pickle.load(open(inputFile, "rb"))
    print("Data Loaded in {0:.2f}".format(time.time() - start))
    return data


cityDistances = loadPickleFile("wi29.p")


TSPGA = GA(cityDistances, 50, 50, 0)
start = time.time()
TSPGA.converge(1000)
print("Elapsed time: {0:.2f} minutes".format((time.time() - start) / 60))
TSPGA.displayGraph()
