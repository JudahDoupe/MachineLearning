import sys
import random
import math

class NormalizedData:
    def __init__(self, fileName, delimiter):
        inFile = open(fileName, "r")

        self.delimiter = delimiter
        self.fileName = fileName
        self.numNonDataItems = 0
        self.numFeatures = 0
        self.numClassifications = 0
        self.documentation = ""
        self.data = []

        lineNum = 0
        for line in inFile:
            if lineNum == 0:
                self.numNonDataItems = int(line)
            elif lineNum == 1:
                self.numFeatures = int(line)
            elif lineNum == 2:
                self.numClassifications = int(line)
            elif lineNum == 3:
                self.documentation = line
            else:
                features = line.split(delimiter)
                [float(i) for i in features]
                self.data.append(features)
            lineNum += 1

        self.normalize()
        random.shuffle(self.data)
        self.inputs = self.inputs()
        self.outputs = self.outputs()

    def normalize(self):
        mins = [sys.maxsize] * self.numFeatures
        maxs = [-sys.maxsize] * self.numFeatures
        means = [0] * self.numFeatures

        for line in self.data:
            for i in range(self.numFeatures):
                value = float(line[i + self.numNonDataItems])

                if value < mins[i]:
                    mins[i] = value
                if value > maxs[i]:
                    maxs[i] = value
                means[i] += value

        for i in range(self.numFeatures):
            means[i] = means[i] / len(self.data)

        for line in self.data:
            for i in range(self.numFeatures):
                value = float(line[i + self.numNonDataItems])

                line[i + self.numNonDataItems] = (value - means[i]) / (maxs[i] - mins[i])

    def inputs(self):
        inputs = []
        for line in self.data:
            inputs.append(line[self.numNonDataItems:self.numFeatures + self.numNonDataItems])
        return inputs

    def outputs(self):
        outputs = []
        for line in self.data:
            output = [0] * self.numClassifications
            output[int(line[-1])] = 1
            outputs.append(output)
        return outputs

    def saveData(self):

        file = open("normailzed_" + self.fileName, "w")
        file.write(str(self.numNonDataItems) + "\n")
        file.write(str(self.numFeatures) + "\n")
        file.write(str(self.numClasifications) + "\n")
        file.write(str(self.documentation))

        for line in self.data:
            i = 0
            for feature in line:
                file.write(str(feature).replace('\n', ' '))
                if i < len(line) - 1:
                    file.write(self.delimiter)
                i = i + 1
            file.write("\n")
