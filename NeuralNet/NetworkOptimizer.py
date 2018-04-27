from NeuralNet.NeuralNetwork import *
import operator
import matplotlib.patches as mpatches
from NeuralNet.NormalizedData import *

class NetworkOptimizer:

    def __init__(self, fileName, fileDelimter=',', debug=False):
        self.data = NormalizedData(fileName, fileDelimiter)
        self.numDataSets = len(self.data.inputs)
        self.trainingIn, self.trainingOut = self.trainingData()
        self.crossValidationIn, self.crossValidationOut = self.crossValidationData()
        self.testingIn, self.testingOut = self.testingData()
        self.debug = debug

    def trainingData(self):
        return self.data.inputs[:math.floor(self.numDataSets * 0.8)], \
               self.data.outputs[:math.floor(self.numDataSets * 0.8)]

    def crossValidationData(self):
        return self.data.inputs[math.floor(self.numDataSets * 0.8):math.floor(self.numDataSets * 0.9)], \
               self.data.outputs[math.floor(self.numDataSets * 0.8):math.floor(self.numDataSets * 0.9)]

    def testingData(self):
        return self.data.inputs[math.floor(self.numDataSets * 0.9):], \
               self.data.outputs[math.floor(self.numDataSets * 0.9):]

    def NFoldCrossValidation(self, N, learningRates, hiddenLayers, nodesPerLayer):
        networkConfigs = []
        totalIters = len(learningRates) * len(hiddenLayers) * len(nodesPerLayer)
        countIter = 1
        for learningRate in learningRates:
            for hiddenLayer in hiddenLayers:
                for nodes in nodesPerLayer:
                    print("Training: {0}th out of {1}".format(countIter, totalIters))
                    net = Network(self.data.numFeatures, self.data.numClassifications, hiddenLayer, nodes, learningRate)
                    net.debug = self.debug
                    net.train(self.trainingIn, self.trainingOut)
                    accuracy = net.test(self.crossValidationIn, self.crossValidationOut)
                    networkConfigs.append(
                        {'hiddenLayers': hiddenLayer, 'nodesPerLayer': nodes, 'learningRate': learningRate,
                         'accuracy': accuracy, 'network': net})
                    countIter += 1

        networkConfigs.sort(key=operator.itemgetter('accuracy'))

        NBestConfigs = networkConfigs[0:N]
        countIter = 1
        for config in NBestConfigs:
            print("Testing: {0}th out of {1}".format(countIter, N))
            config['accuracy'] = config['network'].test(self.testingIn, self.testingOut)
            self.exportCostGraph(config)
            countIter += 1
        return NBestConfigs

    def exportCostGraph(self, config):
        x = np.array(config["network"].costsXAxis)
        y = np.array(config["network"].costs)
        costFigure, costAxes = plt.subplots()


        costAxes.plot(x, y, color='blue')
        costLine = mpatches.Patch(color='blue', label='Cost')
        costFigure.legend(handles=[costLine])
        costAxes.set_xlabel('Training Iterations')
        costAxes.set_ylabel('Average Cost')
        costFigure.suptitle('LearningRate:{0} HiddenLayers:{1} NodesPerLayer:{2}\nAccuracy:{3}'.format(
            config['learningRate'], config['hiddenLayers'], config['nodesPerLayer'], config['accuracy']))
        costFigure.savefig('CostGraphs/LR:{0}_HL:{1}_NPL:{2}.png'.format(
            config['learningRate'], config['hiddenLayers'], config['nodesPerLayer']))

        plt.close(costFigure)

if __name__ == "__main__":
    fileName = 'fishersIris.txt'
    fileDelimiter = ','

    tester = NetworkOptimizer(fileName)
    tester.NFoldCrossValidation(48, [1,.1,.01,.001], [1, 2, 3], [3, 4, 5, 6])
