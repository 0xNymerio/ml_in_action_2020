import operator

# função que calcula a shannon entropy de um dataset
def calcShannonEnt(dataSet):
    from math import log
    numEntries = len(dataSet)
    labelCounts = {}
    # crair um dicionario com todas as classes possiveis
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts:
            labelCounts[currentLabel] = 1
        else:
            labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    for key in labelCounts:
        prob = labelCounts[key] / numEntries
        # Logaritimo base 2
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# Divide o dataset para determinada feature
def splitDataSet(dataSet, axis, value):
    # Cria listas separadas
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            # Cut out the feature split on
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

# Escolhe a melhor feature para o split
def chooseBestFeatureToSplit(dataSet):
    numEntries = len(dataSet)
    numFeatures = len(dataSet[0]) - 1
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain, bestFeature = 0, -1
    for i in range(numFeatures):
        # Cria uma lista unica da classe labels
        uniqueVals = set([example[i] for example in dataSet])
        newEntropy = 0.0
        # Calcula a entropia pra cada split
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            prob = len(subDataSet) / numEntries
            newEntropy += prob * calcShannonEnt(subDataSet)
        infoGain = baseEntropy - newEntropy
        # Determina o melhor Gain
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    return bestFeature

def majorityCnt(classList):
    classCount = {}
    for vote in classList:
        if vote not in classCount:
            classCount[vote] = 1
        else:
            classCount[vote] += 1
    sortedClassCount = sorted(classCount.items(), key=lambda x: x[1],
                              reverse=True)
    return sortedClassCount[0][0]

# Tree-building code
def createTree(dataSet, labels):
    classList = [example[-1] for example in dataSet]
    # Stop when all classes are equal
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # When no more features, return majority
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)
    bestFeat = chooseBestFeatureToSplit(dataSet)
    bestFeatLabel = labels[bestFeat]
    myTree = {bestFeatLabel: {}}
    # Get list of unique values
    del(labels[bestFeat])
    featValues = [example[bestFeat] for example in dataSet]
    for value in set(featValues):
        subLabels = labels[:]
        myTree[bestFeatLabel][value] = createTree(
            splitDataSet(dataSet, bestFeat, value), subLabels)
    return myTree

# Classification function for an existing decision tree
def classify(inputTree, featLabels, testVec):
    firstStr = list(inputTree.keys())[0]
    secondDict = inputTree[firstStr]
    # Translate label string to index
    featIndex = featLabels.index(firstStr)
    key = testVec[featIndex]
    valueOfFeat = secondDict[key]
    if isinstance(valueOfFeat, dict):
        classLabel = classify(valueOfFeat, featLabels, testVec)
    else:
        classLabel = valueOfFeat
    return classLabel

# Methods for persisting the decision tree with pickle
def storeTree(inputTree, filename):
    import pickle
    fw = open(filename, "wb")
    pickle.dump(inputTree, fw)
    fw.close()


def grabTree(filename):
    import pickle
    fr = open(filename, "rb")
    return pickle.load(fr)
