import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class GA:

    def __init__(self, cityNames, cityDistances, populationSize, crossoverRate, mutationRate, twoPointCrossover = True):
        self.cityNames = cityNames
        self.cityDistances = cityDistances

        self.mutationRate = mutationRate
        self.crossoverRate = crossoverRate
        self.twoPointCrossover = twoPointCrossover
        self.populationSize = populationSize

        self.currentGen = []
        self.nextGen = []
        self.generation = 0

        #graph stuff
        self.generations = []
        self.avgFitnesses = []

        for _ in range(self.populationSize):
            self.currentGen.append(path(len(self.cityNames)))

    def converge(self, maxGens = 1000):
        while self.generation < maxGens:
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
        if self.crossoverRate < random.randrange(0, 100):
            return male, female

        idxs = [random.randrange(0, len(male.genes)), random.randrange(0, len(male.genes))]

        #crossover
        if self.twoPointCrossover:
            maleBaby = male.genes[:min(idxs)] + female.genes[min(idxs):max(idxs)] + male.genes[max(idxs):]
            femaleBaby = female.genes[:min(idxs)] + male.genes[min(idxs):max(idxs)] + female.genes[max(idxs):]
        else:
            maleBaby = male.genes[:min(idxs)] + female.genes[min(idxs):]
            femaleBaby = female.genes[:min(idxs)] + male.genes[min(idxs):]

        #repair
        map = []
        for i in range(min(idxs), max(idxs)):
            map.append([male.genes[i], female.genes[i]])

        for i in range(0, min(idxs)):
            for j in range(len(map)):
                if maleBaby[i] == map[j][1]:
                    maleBaby[i] = map[j][0]
                if femaleBaby[i] == map[j][0]:
                    femaleBaby[i] = map[j][1]
        if self.twoPointCrossover:
            for i in range(max(idxs), len(male.genes)):
                for j in range(len(map)):
                    if maleBaby[i] == map[j][1] and maleBaby.count(maleBaby[i]) > 1:
                        maleBaby[i] = map[j][0]
                    if femaleBaby[i] == map[j][0] and femaleBaby.count(femaleBaby[i]) > 1:
                        femaleBaby[i] = map[j][1]

        return path(len(self.cityNames),maleBaby), path(len(self.cityNames),femaleBaby)

    def binaryTournament(self):
        candidate1, candidate2 = self.currentGen[random.randrange(0, len(self.currentGen))], \
                                 self.currentGen[random.randrange(0, len(self.currentGen))]
        if candidate1.fitness(self.cityDistances) > candidate2.fitness(self.cityDistances):
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
        self.genes = []
        for i in range(numCities):
            self.genes.append(i)
        random.shuffle(self.genes)

    def fitness(self, cityDistances):
        sum = 0
        for i in range(len(self.genes)):
            city = self.genes[i]
            nextCity = self.genes[(i+1)%len(self.genes)]
            sum += cityDistances[city][nextCity]
        return sum

    def mutate(self, rate):
        if rate < random.randrange(0, 100):
            pass

        chromosome1 = random.randrange(0, len(self.genes))
        chromosome2 = random.randrange(0, len(self.genes))

        tmp = self.genes[chromosome1]
        self.genes[chromosome1] = self.genes[chromosome2]
        self.genes[chromosome2] = tmp


cityNames =["Houston","Dallas","Austin","Abilene","Waco"]
cityDistances = [[0  ,241,162,351,183],
                 [241,0  ,202,186,97 ],
                 [162,202,0  ,216,106],
                 [351,186,216,0  ,186],
                 [183,97 ,106,186,0  ]]

test = GA(cityNames, cityDistances, 20, 70, 10)
test.converge(25)
test.displayGraph()