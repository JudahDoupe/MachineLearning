""" The following code is meant only for finding a univariate
(one feature) linear regression line given a set of data"""
import time
import numpy as np
import matplotlib.pyplot as plt
global DEBUG
DEBUG=True

#Code the data reading function to produce parallel lists
#xList(our single feature) and yList(our actual y value when
#the parallel x value in xList was observed.  It should be
#noted that the same x value may result in a different observed
#value is our training data set.  We expect to read in m training
#pairs from our file
#Input: the file name
#Outputs: xList, parallel yList, m (the number of #training cases)
def readAndSetUpTrainingValues(fileName):
    inFile=open(fileName,"r")
    xList=[]
    yList=[]
    m=0
    for line in inFile:
        items=line.split(",")
        xList.append(float(items[0]))
        yList.append(float(items[1]))
        m+=1
    return xList,yList,m


#Code the hypothesis function
#Inputs: x value, theta0 and theta1
#Output: predicted value (predicted y)
def hOfX(x,theta0,theta1):
    return theta0+theta1*x


#Code the cost function.
#Not used directly for linear regression (the derivative
#is used to adjust #thetas in our gradient descent algorithm)
#but can be output as an indication that the training process
#is moving in the right direction, or used to terminate
#gradient descent loop
#Invokes hOfX
#Inputs: m, theta0, theta1, xList, yList
#Output: cost value
def JCost(m,theta0,theta1,xList,yList):
    sumSquaredDiffs=0
    for i in range(m):
        sumSquaredDiffs+=(hOfX(xList[i],theta0,theta1)-yList[i])**2
        cost=1/(2*m)*sumSquaredDiffs
    return cost


#Code the partial derivative of JCost (given in notes) for use
#in gradient descent process
#Inputs: m, theta0, theta1, xList, yList
#Outputs: totalDiffs for Theta0 and Theta1
def JCostDerivForGradientDescent(m,theta0,theta1,xList,yList):
    totalDiffsForTheta0=0
    totalDiffsForTheta1=0
    for i in range(m):
        temp=hOfX(xList[i],theta0,theta1)-yList[i]
        totalDiffsForTheta0+=temp
        totalDiffsForTheta1+=temp*xList[i]
    return totalDiffsForTheta0,totalDiffsForTheta1


#Code the gradient descent process loop for convergence on
#the global minimum (for univariate linear regression should
#always be able to find the only minimum that exists for JCost
#if alpha is correctly chosen)
#Inputs: m, xList,yList,alpha,threshold,maxIters
#Outputs: theta0, theta1, (doc purposes: alpha, countIters, threshold)
def gradientDescent(m,xList,yList,alpha,threshold,maxIters):
    countIters=0
    theta0=0
    theta1=0
    currentJCost=JCost(m,theta0,theta1,xList,yList)
    prevJCost=currentJCost+1

    if DEBUG:
       regressionFigure, regressionAxes = plt.subplots()
       costFigure, costAxes = plt.subplots()
       regressionFigure.show()
       costFigure.show()
       x = np.array(xList)
       y = np.array(yList)
       costs = []
       itterations = []

    if DEBUG:print(currentJCost)
    while countIters<maxIters and abs(prevJCost-currentJCost) > threshold:
        tDiffsTheta0,tDiffsTheta1 = JCostDerivForGradientDescent(m,theta0,theta1,xList,yList)
        temp0=theta0 - alpha * ((1/m) * (tDiffsTheta0))
        temp1=theta1 - alpha * ((1/m) * (tDiffsTheta1))
        theta0=temp0
        theta1=temp1
        prevJCost=currentJCost
        currentJCost=JCost(m,theta0,theta1,xList,yList)
        if countIters%100==0  and DEBUG:
            regressionAxes.clear()
            regressionAxes.scatter(x, y)
            regressionAxes.plot(x, theta0 + theta1 * x, color='red')
            regressionFigure.canvas.draw()

            costs.append(currentJCost)
            itterations.append(countIters)
            costAxes.clear()
            costAxes.plot(itterations,costs, color='blue')
            costFigure.canvas.draw()

        countIters+=1
        if DEBUG:print(currentJCost)
    print(tDiffsTheta0,tDiffsTheta1,currentJCost)
    if DEBUG:print("Gradient Descent iterations",countIters)
    return theta0, theta1, alpha, countIters, threshold

def main():
    xList,yList,m=readAndSetUpTrainingValues("versicolorSepalWidthSepalLength.csv")
    if DEBUG:print(xList)
    if DEBUG:print(yList)
    if DEBUG:print(m)
    start=time.time()
    theta0,theta1,alpha,iters,threshold=gradientDescent(m,xList,yList,alpha=.0001,threshold=.000000000001,maxIters=1000000)
    stop=time.time()
    print("Theta0=",theta0)
    print("Theta1=",theta1)
    print("For alpha",format(alpha,'.6f'),"with",iters,"iterations","and threshold",format(threshold,'.18f'),"took",format(stop-start,'.2f'),"seconds")

main()

