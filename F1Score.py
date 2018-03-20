# print("Enter True Positive")
# truePostive = int(input())
#
# print("Enter False Positive")
# falsePostive = int(input())
#
# print("Enter True Negative")
# trueNegative = int(input())
#
# print("Enter False Negative")
# falseNegative = int(input())


truePositive = 100
falsePositive = 0
trueNegative = 9900
falseNegative = 0

cP = truePositive + falseNegative
cN = trueNegative + falsePositive
rP = truePositive + falsePositive
rN = trueNegative + falseNegative

sensitivity = truePositive / cP
specificity = trueNegative / cN
precisionPositive = truePositive / rP
precisionNegative = trueNegative / rN
accuracy = (truePositive + trueNegative) / (cP + cN)
F1 = 2 * ((precisionPositive * sensitivity) / (precisionPositive + sensitivity))

print("Recall:       {}".format(sensitivity))
print("specificity:  {}".format(specificity))
print("precision(+): {}".format(precisionPositive))
print("precision(-): {}".format(precisionNegative))
print("F1:           {}".format(F1))

