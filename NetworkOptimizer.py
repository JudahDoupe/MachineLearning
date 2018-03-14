from NeuralNetwork import *
from NormalizedData import *


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

    def NFoldCrossValidation(self, learningRates, hiddenLayers, nodesPerLayer):
        networkConfigs = []
        totalIters = len(learningRates) * len(hiddenLayers) * len(nodesPerLayer)
        currentIter = 1
        for learningRate in learningRates:
            for hiddenLayer in hiddenLayers:
                for nodes in nodesPerLayer:
                    print("{0}th iteration out {1}".format(currentIter, totalIters))
                    net = Network(self.data.numFeatures, self.data.numClassifications, hiddenLayer, nodes, learningRate)
                    net.debug = self.debug
                    net.train(self.trainingIn, self.trainingOut)
                    accuracy = net.test(self.crossValidationIn, self.crossValidationOut)
                    networkConfigs.append(
                        {'hiddenLayers': hiddenLayer, 'nodesPerLayer': nodes, 'learningRate': learningRate,
                         'accuracy': accuracy})
                    currentIter += 1
        bestConfig = None
        accuracy = 0
        for i in range(0, len(networkConfigs)):
            tempAccuracy = networkConfigs[i]['accuracy']

            if tempAccuracy > accuracy:
                accuracy = tempAccuracy
                bestConfig = networkConfigs[i]

        net = Network(self.data.numFeatures, self.data.numClassifications, bestConfig['hiddenLayers'],
                      bestConfig['nodesPerLayer'], bestConfig['learningRate'])
        net.train(self.testingIn, self.testingOut)
        bestConfig['accuracy'] = net.test(self.testingIn, self.testingOut)
        return bestConfig


if __name__ == "__main__":
    fileName = 'fishersIris.txt'
    fileDelimiter = ','

    tester = NetworkOptimizer(fileName)

    print(tester.NFoldCrossValidation([1], [1, 2], [3, 4, 5, 6]))
