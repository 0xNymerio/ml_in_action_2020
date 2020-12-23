import numpy as np
import matplotlib.pyplot as plt

def sigmoid(inX):
    return 1 / (1 + np.exp(-inX))

def stocGradAscent1(dataMatrix, classLabels, numIter=150):
    m, n = dataMatrix.shape
    weights = np.ones(n)
    for j in range(numIter):
        dataIndex = list(range(m))
        for i in range(m):
            alpha = 4 / (1.0 + j + i) + 0.01
            randIndex = int(np.random.uniform(0, len(dataIndex)))
            h = sigmoid(np.dot(dataMatrix[dataIndex[randIndex]], weights))
            error = classLabels[dataIndex[randIndex]] - h
            weights = (weights +
                       alpha * error * dataMatrix[dataIndex[randIndex]])
            del(dataIndex[randIndex])
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(np.dot(inX, weights))
    if prob > 0.5:
        return 1
    else:
        return 0


def colicTest():
    frTrain = open('./logistic_regression/horseColicTraining.txt')
    frTest = open('./logistic_regression/horseColicTest.txt')
    trainingSet, trainingLabels = [], []
    for line in frTrain.readlines():
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        trainingSet.append(lineArr)
        trainingLabels.append(float(currLine[21]))
    trainWeights = stocGradAscent1(np.array(trainingSet),
                                   np.array(trainingLabels), 1000)
    errorCount, numTestVec = 0, 0
    for line in frTest.readlines():
        numTestVec += 1
        currLine = line.strip().split('\t')
        lineArr = []
        for i in range(21):
            lineArr.append(float(currLine[i]))
        if int(classifyVector(np.array(lineArr),
                              trainWeights)) != int(currLine[21]):
            errorCount += 1
    errorRate = errorCount / numTestVec
    print("the error rate of this test is: %f" % errorRate)
    return errorRate


def multiTest():
    numTests = 10
    errorSum = 0
    for k in range(numTests):
        errorSum += colicTest()
    print("after %d iterations the average error rate is: %f"
          % (numTests, errorSum / numTests))

multiTest()
