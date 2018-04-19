import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

class GA:

    def __init__(self, cityNames, cityDistances, crossoverRate, mutationRate, twoPointCrossover = True):
        self.cityNames = cityNames
        self.cityDistances = cityDistances

        self.currentGen = []
        self.nextGen = []
        self.generation = 0
        self.generations = []
        self.avgFitness = 0
        self.avgFitnesses = []
        self.fitnesses = [[], []]
        self.drawAllFitnesses = False

        self.mutationRate = mutationRate
        self.crossoverRate = crossoverRate
        self.twoPointCrossover = twoPointCrossover
        self.populationSize = len(cityNames)


        for _ in range(self.populationSize):
            self.currentGen.append(path())

    def converge(self, fitnessPercent, maxGens = 10000):
        while self.avgFitness < (fitnessPercent * int("1111111111111111", 2) * int("1111111111111111", 2)) or self.generation < maxGens:
            self.avgFitness = 0
            for _ in range(len(self.currentGen) // 2):
                male = self.binaryTournament()
                female = self.binaryTournament()

                baby1, baby2 = self.crossover(male, female)

                baby1.mutate(self.mutationRate)
                baby2.mutate(self.mutationRate)

                self.nextGen.append(baby1)
                self.nextGen.append(baby2)
                self.avgFitness += baby1.fitness()
                self.avgFitness += baby2.fitness()
                self.fitnesses[0].append(self.generation)
                self.fitnesses[1].append(baby2.fitness())
                self.fitnesses[0].append(self.generation)
                self.fitnesses[1].append(baby2.fitness())

            self.avgFitness = self.avgFitness // len(self.nextGen)

            self.currentGen = self.nextGen
            self.nextGen = []

            self.generation += 1
            self.generations.append(self.generation)
            self.avgFitnesses.append(self.avgFitness)

    def crossover(self, male, female):
        if self.crossoverRate < random.randrange(0, 100):
            return male, female

        idxs = [random.randrange(0, 32), random.randrange(0, 32)]

        if self.twoPointCrossover:
            baby1 = male.genes[:min(idxs)] + female.genes[min(idxs):max(idxs)] + male.genes[max(idxs):]
            baby2 = female.genes[:min(idxs)] + male.genes[min(idxs):max(idxs)] + female.genes[max(idxs):]
        else:
            baby1 = male.genes[:min(idxs)] + female.genes[min(idxs):]
            baby2 = female.genes[:min(idxs)] + male.genes[min(idxs):]

        return path(baby1), path(baby2)

    def binaryTournament(self):
        candidate1, candidate2 = self.currentGen[random.randrange(0, len(self.currentGen))], \
                                 self.currentGen[random.randrange(0, len(self.currentGen))]
        if candidate1.fitness() > candidate2.fitness():
            winner = candidate1
        else:
            winner = candidate2

        return winner

    def FitnessGraph(self):
        x = np.array(self.generations)
        y = np.array(self.avgFitnesses)
        fig, axes = plt.subplots()
        axes.set_ylim([-int("1111111111111111",2) * int("1111111111111111",2), int("1111111111111111",2) * int("1111111111111111",2)])
        axes.plot(x, y, color='blue')
        if self.drawAllFitnesses:
            axes.scatter(self.fitnesses[0],self.fitnesses[1])
        fitnessPatch = mpatches.Patch(color='blue', label='Avg Fitness')
        fig.legend(handles=[fitnessPatch])
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
            self.genes = self.initGenes(numCities)
        else:
            self.genes = genes

    def initGenes(self, numCities):
        genes = []
        for i in range(numCities):
            genes.append(i)
        return random.shuffle(genes)

    def fitness(self, cityDistances):
        sum = 0
        for i in range(len(self.genes)):
            sum += cityDistances[i][(i+1)%len(self.genes)]
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

test = GA(cityNames, cityDistances, 70, 10)
test.converge(0.95)