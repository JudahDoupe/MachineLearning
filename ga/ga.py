import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

mutationRate = 10
crossoverRate = 70
twoPointCrossover = False
populationSize = 50


drawAllFitnesses = False


class individual:

    def __init__(self, genes=None, ):
        if not genes:
            self.genes = self.initGenes()
        else:
            self.genes = genes

    def initGenes(self):
        string = ""
        for _ in range(32):
            string = string + str(random.randrange(0, 2))
        return string

    def fitness(self):
        x = int(self.genes[:16], 2)
        y = int(self.genes[16:], 2)
        return (x * x) - (y * y)

    def mutate(self):
        if mutationRate < random.randrange(0, 100):
            pass

        chromosome = random.randrange(0, 32)
        chromosomes = list(self.genes)

        if self.genes[chromosome] == '1':
            chromosomes[chromosome] = '0'
            self.genes = ''.join(chromosomes)
        else:
            chromosomes[chromosome] = '1'
            self.genes = ''.join(chromosomes)


def crossover(male, female, twoPoint = False):
    if crossoverRate < random.randrange(0, 100):
        return male, female

    idxs = [random.randrange(0, 32),random.randrange(0, 32)]

    if twoPoint:
        baby1 = male.genes[:min(idxs)] + female.genes[min(idxs):max(idxs)] + male.genes[max(idxs):]
        baby2 = female.genes[:min(idxs)] + male.genes[min(idxs):max(idxs)] + female.genes[max(idxs):]
    else:
        baby1 = male.genes[:min(idxs)] + female.genes[min(idxs):]
        baby2 = female.genes[:min(idxs)] + male.genes[min(idxs):]

    return individual(baby1), individual(baby2)


def initPopulation(size):
    population = []
    for _ in range(size):
        population.append(individual())
    return population



def binaryTournament(oldGen):
    candidate1, candidate2 = oldGen[random.randrange(0, len(oldGen))], \
                             oldGen[random.randrange(0, len(oldGen))]
    if candidate1.fitness() > candidate2.fitness():
        winner = candidate1
    else:
        winner = candidate2

    return winner


def FitnessGraph():
    x = np.array(generations)
    y = np.array(avgFitnesses)
    fig, axes = plt.subplots()
    axes.set_ylim([-int("1111111111111111",2) * int("1111111111111111",2), int("1111111111111111",2) * int("1111111111111111",2)])
    axes.plot(x, y, color='blue')
    if drawAllFitnesses:
        axes.scatter(fitnesses[0],fitnesses[1])
    fitnessPatch = mpatches.Patch(color='blue', label='Avg Fitness')
    fig.legend(handles=[fitnessPatch])
    axes.set_xlabel('Generations')
    axes.set_ylabel('Fitness')
    fig.suptitle('GA')
    #fig.savefig('GA.png')
    fig.show()
    #plt.close(fig)


oldGen = initPopulation(populationSize)
newGen = []




avgFitness = 0
generation = 0
generations = []
avgFitnesses = []
fitnesses = [[],[]]








while avgFitness < (0.95 * int("1111111111111111",2) * int("1111111111111111",2)):


    avgFitness = 0

    for _ in range(len(oldGen)//2):
        male = binaryTournament(oldGen)
        female = binaryTournament(oldGen)

        baby1, baby2 = crossover(male, female,twoPointCrossover)

        baby1.mutate()
        baby2.mutate()

        newGen.append(baby1)
        newGen.append(baby2)
        avgFitness += baby1.fitness()
        avgFitness += baby2.fitness()
        fitnesses[0].append(generation)
        fitnesses[1].append(baby2.fitness())
        fitnesses[0].append(generation)
        fitnesses[1].append(baby2.fitness())


    avgFitness = avgFitness//len(newGen)


    oldGen = newGen
    newGen = []

    generation += 1
    generations.append(generation)
    avgFitnesses.append(avgFitness)

print("Generations to convergence {0}".format(generation))
FitnessGraph()
input("Press enter to close graph: ")