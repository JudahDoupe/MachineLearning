import random
import math
from NormalizedData import *


class Node:
    def __init__(self, layer):
        self.layer = layer
        self.connections = {}
        self.bias = random.uniform(-0.5, 0.5)
        self.output = None
        self.errDrv = None

    def calcOutput(self):
        if self.layer.isInput():
            return self.output

        netInput = self.bias
        for node in self.layer.previousLayer.nodes:
            netInput += node.output * node.connections[self]

        self.output = 1 / (1 + math.exp(-netInput))  # sigmoid function
        return self.output

    def calcErrDrv(self):
        if self.layer.isOutput():
            return None

        errDrvSum = 0
        for node, weight in self.connections.items():
            errDrvSum += node.errDrv * weight

        self.errDrv = self.output * (1 - self.output) * errDrvSum
        return self.errDrv

    def calcTrueErrDrv(self, trueValue):
        self.errDrv = self.output * (1 - self.output) * (trueValue - self.output)
        return self.errDrv

    def updateWeights(self):
        self.bias += self.errDrv
        for node, weight in self.connections.items():
            self.connections[node] = weight + node.errDrv * self.output

    def visualize(self):
        if self.errDrv == None:
            self.errDrv = 0
        print("Output: {0:.2f}, Bias: {1:.2f}, ErrDrv {2:.2f}".format(self.output, self.bias, self.errDrv))


class Layer:
    def __init__(self, numNodes):
        self.nodes = []
        for i in range(numNodes):
            self.nodes.append(Node(self))
        self.nextLayer = None
        self.previousLayer = None

    def connect(self, nextLayer):
        self.nextLayer = nextLayer
        nextLayer.previousLayer = self
        for node in self.nodes:
            for nextNode in nextLayer.nodes:
                node.connections[nextNode] = random.uniform(-0.5, 0.5)

    def calcOutputs(self):
        if self.isInput():
            return
        for node in self.nodes:
            node.calcOutput()

    def calcErrDrv(self):
        if self.isOutput():
            return
        for node in self.nodes:
            node.calcErrDrv()

    def updateWeights(self):
        for node in self.nodes:
            node.updateWeights()

    def isOutput(self):
        return self.nextLayer == None

    def isInput(self):
        return self.previousLayer == None

    def index(self):
        if self.previousLayer == None:
            return 0
        else:
            return self.previousLayer.index() + 1

    def visualize(self):
        print("Layer {} :".format(self.index()))
        for node in self.nodes:
            node.visualize()


class Network:
    def __init__(self, numFeatures, numClassifications, numHiddenLayers, numHiddenNodes):
        self.debug = False
        self.layers = []
        # input layer
        self.addLayer(numFeatures)
        # hidden layers
        for i in range(numHiddenLayers):
            self.addLayer(numHiddenNodes)
        # output layer
        self.addLayer(numClassifications)

    def addLayer(self, numNodes):
        newLayer = Layer(numNodes)
        if len(self.layers) > 0:
            prevLayer = self.layers[-1]
            prevLayer.connect(newLayer)
        self.layers.append(newLayer)

    def train(self, inputs, outputs):
        for i in range(0, len(inputs)):
            self.forwardProp(inputs[i])
            self.backProp(outputs[i])
        if self.debug:
            self.visualize()

    def test(self, inputs, outputs):
        numSuccesses = 0
        for i in range(0, len(inputs)):
            results = self.forwardProp(inputs[i])
            if results.index(max(results)) == outputs[i].index(max(outputs[i])):
                numSuccesses += 1
        print("Accuracy:   {0:.1f}%   ".format(numSuccesses / len(inputs) * 100))

    def forwardProp(self, inputs):
        for i in range(len(inputs)):
            self.layers[0].nodes[i].output = inputs[i]

        for i in range(1, len(self.layers)):
            self.layers[i].calcOutputs()

        output = []
        for i in range(len(self.layers[-1].nodes)):
            output.append(self.layers[-1].nodes[i].output)
        return output

    def backProp(self, outputs):
        for i in range(len(outputs)):
            self.layers[-1].nodes[i].calcTrueErrDrv(outputs[i])
        self.layers[-1].updateWeights()

        for i in range(len(self.layers) - 2, -1, -1):
            self.layers[i].calcErrDrv()
            self.layers[i].updateWeights()

    def visualize(self):
        for layer in self.layers:
            layer.visualize()
        print()


if __name__ == "__main__":

    trainingFile = input("Training Data Filename: ")
    trainingDelimiter = input("Delimiter: ")
    trainingData = NormalizedData(trainingFile, trainingDelimiter)
    
    testingFile = input("Testing Data Filename: ")
    testingDelimiter = input("Delimiter: ")
    testingData = NormalizedData(testingFile, testingDelimiter)

    net = Network(trainingData.numFeatures,
                  trainingData.numClassifications,
                  1,
                  math.floor(trainingData.numFeatures * 1.5))
    
    trainingInputs = trainingData.inputs()
    trainingOutputs = trainingData.outputs()

    testingInputs = testingData.inputs()
    testingOutputs = testingData.outputs()

    for i in range(0, 1):
        net.train(trainingInputs, trainingOutputs)

    net.test(testingInputs, testingOutputs)
