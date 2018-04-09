print("This take in 4 values: True Positive, False Positive, True Negative, False Negative.")
print("Enter True Positive: ", end='')
truePositive = int(input())

print("Enter False Positive: ", end='')
falsePositive = int(input())

print("Enter True Negative: ", end='')
trueNegative = int(input())

print("Enter False Negative: ", end='')
falseNegative = int(input())


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

