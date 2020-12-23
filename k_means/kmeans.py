import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.datasets import load_wine
import pandas as pd

def distSLC(vecA, vecB):
    a = np.sin(vecA[1] * np.pi / 180) * np.sin(vecB[1] * np.pi / 180)
    b = (np.cos(vecA[1] * np.pi / 180) * np.cos(vecB[1] * np.pi / 180) *
         np.cos(np.pi * (vecB[0] - vecA[0]) / 180))
    return np.arccos(a + b) * 6371.0

def distEclud(vecA, vecB):
    return np.sqrt(np.sum(np.power(vecA - vecB, 2)))

def randCent(dataSet, k):
    n = dataSet.shape[1]
    centroids = np.zeros((k, n))
    # Criando os centroides
    for j in range(n):
        minJ = min(dataSet[:, j])
        rangeJ = max(dataSet[:, j]) - minJ
        centroids[:, j] = minJ + rangeJ * np.random.rand(k)
    return centroids

def kMeans(dataSet, k, distMeas=distEclud, createCent=randCent):
    m = dataSet.shape[0]
    clusterAssment = np.zeros((m, 2))
    centroids = createCent(dataSet, k)
    clusterChanged = True
    while clusterChanged:
        clusterChanged = False
        for i in range(m):
            minDist = np.inf
            minIndex = -1
            # Procura o centroide mais proximo
            for j in range(k):
                distJI = distMeas(centroids[j], dataSet[i])
                if distJI < minDist:
                    minDist = distJI
                    minIndex = j
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i] = [minIndex, minDist ** 2]
        for cent in range(k):
            ptsInClust = dataSet[np.nonzero(clusterAssment[:, 0] == cent)[0]]
            centroids[cent] = np.mean(ptsInClust, axis=0)
    return centroids, clusterAssment

def biKmeans(dataSet, k, distMeas=distEclud):
    m = dataSet.shape[0]
    clusterAssment = np.zeros((m, 2))
    # Cria o cluster
    centList = [np.mean(dataSet, axis=0)]
    for j in range(m):
        clusterAssment[j, 1] = distMeas(centList[0], dataSet[j]) ** 2
    while len(centList) < k:
        lowestSSE = np.inf
        for i in range(len(centList)):
            # Divide cada Cluster
            ptsInCurrCluster = dataSet[np.nonzero(clusterAssment[:, 0] == i)[0]]
            centroidMat, splitClustAss = kMeans(ptsInCurrCluster, 2, distMeas)
            sseSplit = sum(splitClustAss[:, 1])
            sseNotSplit = sum(
                clusterAssment[np.nonzero(clusterAssment[:, 0] != i)[0], 1])
            if sseSplit + sseNotSplit < lowestSSE:
                bestCentToSplit = i
                bestNewCents = centroidMat
                bestClustAss = splitClustAss.copy()
                lowestSSE = sseSplit + sseNotSplit
        bestClustAss[
            np.nonzero(bestClustAss[:, 0] == 1)[0], 0] = len(centList)
        bestClustAss[
            np.nonzero(bestClustAss[:, 0] == 0)[0], 0] = bestCentToSplit
        centList[bestCentToSplit] = bestNewCents[0]
        centList.append(bestNewCents[1])
        clusterAssment[np.nonzero(
            clusterAssment[:, 0] == bestCentToSplit)[0]] = bestClustAss
    return np.array(centList), clusterAssment


def cluster(numClust=3):
    df = pd.read_csv('http://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv',sep=';') #header=0
    print(df)
    df = df[:100]



    fixed_acidity__citric_acid = df[['fixed acidity','citric acid']].values
    fixed_acidity__density = df[['fixed acidity','density']].values
    total_sulfur_dioxide__ph= df[['total sulfur dioxide','pH']].values
    fixed_acidity__ph= df[['fixed acidity','pH']].values

    datMat = total_sulfur_dioxide__ph

    myCentroids, clustAssing = biKmeans(datMat, numClust, distMeas=distSLC) 	#distEclud e distSLC
    #plt.figure()
    scatterMarkers = ['s', 'o', '^', '8', 'p', 'd', 'v', 'h', '>', '<']
    # Create matrix from image
    plt.figure(figsize=(5, 4))
    plt.xlabel("Total Sulfur Dioxide")
    plt.ylabel("pH")
    plt.title(" Total Sulfur Dioxide e pH - 3 Centroides")
    for i in range(numClust):
        ptsInCurrCluster = datMat[np.nonzero(clustAssing[:, 0] == i)[0]]
        markerStyle = scatterMarkers[i % len(scatterMarkers)]
        plt.scatter(ptsInCurrCluster[:, 0], ptsInCurrCluster[:, 1],
                    marker=markerStyle, s=90)
    plt.scatter(myCentroids[:, 0], myCentroids[:, 1],
                marker='+', s=300)


    plt.show()

cluster(3)
