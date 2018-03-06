import sys

DELIMITER = ","
FILENAME = "dummyData.txt"


def ParseData(fileName):
	inFile=open(fileName,"r")

	numNonDataItems = 0
	numFeatures = 0
	numClasifications = 0
	documentation = ""
	data = []

	lineNum = 0
	for line in inFile:
		if lineNum == 0:
			numNonDataItems = int(line)
		elif lineNum == 1:
			numFeatures = int(line)
		elif lineNum == 2:
			numClasifications = int(line)
		elif lineNum == 3:
			documentation = line
		else:
			features=line.split(DELIMITER)
			data.append(features)

		lineNum += 1

	return numNonDataItems, numFeatures, numClasifications, documentation, data

def normailze(data, numNonDataItems, numFeatures):

	mins = [sys.maxint] * numFeatures
	maxs = [-sys.maxint] * numFeatures
	means = [0] * numFeatures

	for line in data:
		for  i in range(numFeatures):
			value = float(line[i+numNonDataItems])

			if value< mins[i]:
				mins[i] = value
			if value > maxs[i]:
				maxs[i] = value
			means[i] += value

	for i in range(numFeatures):
		means[i] = means[i] / len(data)

	for line in data:
		for  i in range(numFeatures):
			value = float(line[i+numNonDataItems])

			line[i+numNonDataItems] = (value - means[i]) / (maxs[i] - mins[i])

	return data

def saveData(numNonDataItems, numFeatures, numClasifications, documentation, data):

	file = open("normailzed"+FILENAME,"w")
	file.write(str(numNonDataItems)+"\n")
	file.write(str(numFeatures)+"\n")
	file.write(str(numClasifications)+"\n")
	file.write(str(documentation))

	for line in data:
		i = 0
		for feature in line:
			file.write(str(feature).replace('\n',' '))
			if i  < len(line)-1:
				file.write(DELIMITER)
			i = i+1
		file.write("\n")


def main():
	numNonDataItems, numFeatures, numClasifications, documentation, data = ParseData(FILENAME)
	normalizedData = normailze(data, numNonDataItems, numFeatures)
	saveData(numNonDataItems, numFeatures, numClasifications, documentation, normalizedData)


main()
