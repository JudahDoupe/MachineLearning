import random
import math
class node:
    def __init__(self, value=None):
        self.theta = None
        self.connections = {}
        self.FFvalue = value
        self.layer = None
        self.errDrv = None
    def calcFFvalue(self):
        netInput = 0
        for node in self.layer.previousLayer.nodes:
            netInput += node.FFvalue * node.connections[self]
        netInput += self.theta
        self.FFvalue = self.sigmoid(netInput)
        return self.FFvalue
    def updateThetas(self):
        self.theta = self.theta + self.errDrv
        for node, weight in self.connections.items():
            self.connections[node] = weight + node.errDrv * self.FFvalue
    def calcOutputErrDrv(self, trueValue):
        self.errDrv = self.FFvalue * (1 - self.FFvalue) * (trueValue - self.FFvalue)
        return self.errDrv
    def calcErrDrv(self):
        errDrvSum = 0
        for node, weight in self.connections.items():
            errDrvSum += node.errDrv * weight
        self.errDrv = self.FFvalue * (1 - self.FFvalue) * errDrvSum
        return self.errDrv
    def sigmoid(self, netInput):
        return 1 / (1 + math.exp(-netInput))
class layer:
    def __init__(self, nodes, nextLayer):
        self.nodes = nodes
        self.nextLayer = nextLayer
        self.previousLayer = None
        for node in nodes:
            node.layer = self
            node.theta = random.uniform(-0.5, 0.5)
            if not self.isOutput():
                for nextNode in nextLayer.nodes:
                    node.connections[nextNode] = random.uniform(-0.5, 0.5)
    def isOutput(self):
        return self.nextLayer == None
    def isInput(self):
        return self.previousLayer == None
    def updateThetas(self):
        for node in self.nodes:
            node.updateThetas()
    def calcErrDrv(self):
        if self.isOutput():
            return
        for node in self.nodes:
            node.calcErrDrv()
    def calcFFvalues(self):
        if self.isInput():
            return
        for node in self.nodes:
            node.calcFFvalue()
class network:
    def __init__(self):
        self.outputLayer = layer([node(), node()], None)
        self.hiddenLayer = layer([node(), node()], self.outputLayer)
        self.outputLayer.previousLayer = self.hiddenLayer
        self.inputLayer = layer([node(), node()], self.hiddenLayer)
        self.hiddenLayer.previousLayer = self.inputLayer
    def forwardProp(self, inputs):
        self.inputLayer.nodes[0].FFvalue = inputs[0]
        self.inputLayer.nodes[1].FFvalue = inputs[1]
        self.hiddenLayer.calcFFvalues()
        self.outputLayer.calcFFvalues()
        return [self.outputLayer.nodes[0].FFvalue, self.outputLayer.nodes[1].FFvalue]
    def backProp(self, outputs):
        self.outputLayer.nodes[0].calcOutputErrDrv(outputs[0])
        self.outputLayer.nodes[1].calcOutputErrDrv(outputs[1])
        self.outputLayer.updateThetas()
        self.hiddenLayer.calcErrDrv()
        self.hiddenLayer.updateThetas()
        self.inputLayer.calcErrDrv()
        self.inputLayer.updateThetas()
    def train(self, inputs, outputs):
        for i in range(0, len(inputs)):
            self.forwardProp(inputs[i])
            self.backProp(outputs[i])
if __name__ == "__main__":
    net = network()
    inputs = [[0, 0], [0, 1], [1, 0], [1, 1]]
    outputs = [[0, 1], [1, 0], [1, 0], [0, 1]]
    for i in range(0, 1000):
        net.train(inputs, outputs)
    for i in range(0, len(inputs)):
        results = net.forwardProp(inputs[i])
        print("result:   {1:.1f}% {0}  ".format(results[0] > results[1], 100*max(results)))
        print("expected: {0}".format(outputs[i][0] > outputs[i][1]))
        print()
