import re
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Standard regression 
def standRegres(xArr, yArr):
    xTx = np.dot(xArr.T, xArr)
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = np.dot(np.linalg.inv(xTx),
                np.dot(xArr.T, yArr[:, np.newaxis]))
    return ws.ravel()

def lwlr(testPoint, xArr, yArr, k=1.0):
    m = xArr.shape[0]
    weights = np.eye(m)
    for j in range(m):
        diffMat = testPoint - xArr[j]
        weights[j, j] = np.exp(np.dot(diffMat, diffMat) / (-2 * k ** 2))
    xTx = np.dot(xArr.T, np.dot(weights, xArr))
    if np.linalg.det(xTx) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = np.dot(np.linalg.inv(xTx),
                np.dot(xArr.T, np.dot(weights, yArr[:, np.newaxis])))
    return np.dot(testPoint, ws.ravel())


def lwlrTest(testArr, xArr, yArr, k=1.0):
    m = testArr.shape[0]
    yHat = np.zeros(m)
    for i in range(m):
        yHat[i] = lwlr(testArr[i], xArr, yArr, k)
    return yHat

def rssError(yArr, yHatArr):
    return ((yArr - yHatArr) ** 2).sum()



# Ridge regression
def ridgeRegres(xArr, yArr, lam=0.2):
    xTx = np.dot(xArr.T, xArr)
    denom = xTx + np.eye(xArr.shape[1]) * lam
    if np.linalg.det(denom) == 0.0:
        print("This matrix is singular, cannot do inverse")
        return
    ws = np.dot(np.linalg.inv(denom),
                np.dot(xArr.T, yArr[:, np.newaxis]))
    return ws.ravel()


def ridgeTest(xArr, yArr):
    yMean = np.mean(yArr)
    yArr = yArr - yMean
    xMeans = np.mean(xArr, axis=0)
    xStd = np.std(xArr, axis=0)
    xArr = (xArr - xMeans) / xStd
    numTestPts = 30
    wMat = np.zeros((numTestPts, xArr.shape[1]))
    for i in range(numTestPts):
        ws = ridgeRegres(xArr, yArr, np.exp(i - 10))
        wMat[i] = ws
    return wMat


def searchForSet(retX, retY, setNum, yr, numPce, origPrc):
    file_name = "./linear_regression/setHtml/lego" + str(setNum) + ".html"
    input_file = open(file_name, encoding="utf8")
    page = input_file.read()
    input_file.close()
    pattern = re.compile(r'\$([\d,]+\.\d+)')
    result = pattern.findall(page)
    for price in result:
        price = price.replace(",", "")
        if float(price) <= origPrc * 0.5:
            continue
        retX.append([yr, numPce, origPrc])
        retY.append(float(price))

def setDataCollect(retX, retY):
    searchForSet(retX, retY, 8288, 2006, 800, 49.99)
    searchForSet(retX, retY, 10030, 2002, 3096, 269.99)
    searchForSet(retX, retY, 10179, 2007, 5195, 499.99)
    searchForSet(retX, retY, 10181, 2007, 3428, 199.99)
    searchForSet(retX, retY, 10189, 2008, 5922, 299.99)
    searchForSet(retX, retY, 10196, 2009, 3263, 249.99)


def crossValidation(xArr, yArr, numVal=10):
    m = len(yArr)
    indexList = list(range(m))
    errorMat = np.zeros((numVal, 30))
    for i in range(numVal):

        trainX, trainY = [], []
        testX, testY = [], []
        np.random.shuffle(indexList)
        for j in range(m):

            if j < m * 0.9:
                trainX.append(xArr[indexList[j]])
                trainY.append(yArr[indexList[j]])
            else:
                testX.append(xArr[indexList[j]])
                testY.append(yArr[indexList[j]])
        trainX, trainY = np.array(trainX), np.array(trainY)
        testX, testY = np.array(testX), np.array(testY)
        wMat = ridgeTest(trainX, trainY)

        meanTrain = np.mean(trainX, axis=0)
        stdTrain = np.std(trainX, axis=0)
        testX = (testX - meanTrain) / stdTrain
        for k in range(30):
            yEst = np.dot(testX, wMat[k]) + np.mean(trainY)
            errorMat[i, k] = rssError(yEst, testY)
    meanErrors = np.mean(errorMat, axis=0)
    bestWeights = wMat[np.argmin(meanErrors)]
    meanX = np.mean(xArr, axis=0)
    stdX = np.std(xArr, axis=0)
    unReg = bestWeights / stdX
    print("the best model from Ridge Regression is:\n", unReg)
    print("with constant term: ",
          -1 * np.dot(meanX, unReg) + np.mean(yArr))

lgX, lgY = [], []
setDataCollect(lgX, lgY)
lgX, lgY = np.array(lgX), np.array(lgY)
lgX1 = np.ones((82, 4))
lgX1[:, 1: 4] = lgX
ws = standRegres(lgX1, lgY)
print(ws)

print(crossValidation(lgX, lgY, 10))
