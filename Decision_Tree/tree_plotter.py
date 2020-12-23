import matplotlib.pyplot as plt

# Plotting tree nodes with text annotations
# Define box and arrow formatting
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")


# Draws annotations with arrows
def plotNode(nodeTxt, centerPt, parentPt, nodeType, ax):
    ax.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction',
                xytext=centerPt, textcoords='axes fraction',
                va="center", ha="center", bbox=nodeType,
                arrowprops=arrow_args)


# The plotTree function
def plotMidText(cntrPt, parentPt, txtString, ax):
    xMid = (parentPt[0] - cntrPt[0]) / 2.0 + cntrPt[0]
    yMid = (parentPt[1] - cntrPt[1]) / 2.0 + cntrPt[1]
    ax.text(xMid, yMid, txtString, va="center", ha="center",
            rotation=30)


def plotTree(myTree, parentPt, nodeTxt, totalW, totalD, xyOff, ax):
    numLeafs = getNumLeafs(myTree)
    firstStr = list(myTree.keys())[0]
    cntrPt = (xyOff[0] + (1 + numLeafs) / (2 * totalW), xyOff[1])
    plotMidText(cntrPt, parentPt, nodeTxt, ax)
    plotNode(firstStr, cntrPt, parentPt, decisionNode, ax)
    secondDict = myTree[firstStr]
    xyOff[1] -= 1 / totalD
    for key in sorted(secondDict.keys()):
        if isinstance(secondDict[key], dict):
            plotTree(secondDict[key], cntrPt, str(key),
                     totalW, totalD, xyOff, ax)
        else:
            xyOff[0] += 1.0 / totalW
            plotNode(secondDict[key], (xyOff[0], xyOff[1]), cntrPt,
                     leafNode, ax)
            plotMidText((xyOff[0], xyOff[1]), cntrPt, str(key), ax)
    xyOff[1] += 1 / totalD


def createPlot(inTree, figsize):
    fig, ax = plt.subplots(subplot_kw={'xticks':[], 'yticks':[],
                                       'frameon':False},
                           figsize=figsize)
    totalW = getNumLeafs(inTree)
    totalD = getTreeDepth(inTree)
    xyOff = [-0.5 / totalW, 1]
    plotTree(inTree, (0.5, 1.0), '', totalW, totalD, xyOff, ax)
    plt.show()

# Conta o numero de leafs e retorna a qtd
def getNumLeafs(myTree):
    numLeafs = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        # Test if node is dictionary
        if isinstance(secondDict[key], dict):
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

# Conta o numero de vezes que você acertou um node de decisão
def getTreeDepth(myTree):
    maxDepth = 0
    firstStr = list(myTree.keys())[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if isinstance(secondDict[key], dict):
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth
