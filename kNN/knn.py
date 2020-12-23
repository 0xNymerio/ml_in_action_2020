# coding=utf-8
from numpy import *
import numpy as np
import operator
import matplotlib.pyplot as plt
import pandas as pd

def calcDistance(inX, dataSet, labels, k):
    # .shape retorna as dimensoes de um array numpy. y.shape() retorna n,m e .shape[0] apenas n (linhas)
    dataSetSize = dataSet.shape[0]
    # numpy.tile (A, reps): constrói uma matriz repetindo A o número de vezes dado por reps
    diffMat = tile(inX, (dataSetSize, 1)) - dataSet
    sqDiffMat = diffMat ** 2
    # sqDistances = x^2 + y^2
    sqDistances = sqDiffMat.sum(axis=1)
    # a distância é igual à raiz quadrada da soma dos quadrados das coordenadas
    distances = sqDistances ** 0.5
    # numpy.argsort () retorna os índices que ordenariam uma matriz
    sortedDistIndices = distances.argsort()
    return sortedDistIndices


def findMajorityClass(inX, dataSet, labels, k, sortedDistIndices):
    classCount = {}
    # itere k vezes do item mais próximo
    for i in range(k):
        voteIlabel = labels[sortedDistIndices[i]]
        # aumenta em 1 no label selecionado
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # No Python 3 não funciona mais o dict.iteritems, agora só o dict.items()
    return sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)


def classify0(inX, dataSet, labels, k):
    # Preferi separar o classifer para minimizar erros, recomendado por alguns forums
    # calcula a distancia entre inX e o ponto atual
    sortedDistIndices = calcDistance(inX, dataSet, labels, k)
    # Pega k itens com distâncias mais baixas para inX e encontre a classe majoritária entre os k itens
    sortedClassCount = findMajorityClass(inX, dataSet, labels, k, sortedDistIndices)
    return sortedClassCount[0][0]

def autoNorm(dataSet):
    #normaliza o sataset a fim de separar ele entre testes e treino.
    # essa porcentagem é colocada em outro momento.
    minVals = dataSet.min(axis=0)
    maxVals = dataSet.max(axis=0)
    ranges = maxVals - minVals
    normDataSet = np.zeros(dataSet.shape)
    normDataSet = dataSet - minVals
    normDataSet = normDataSet / ranges
    return normDataSet, ranges, minVals

def file2matrix(filename):
    # Apelei pro Pandas
    # X é a Matriz com os dados e Y os Labels para classificação
    df = pd.read_csv(filename, header=0)
    x = df.drop(['species'],axis=1).values
    y = df['species'].values
    return x, y


# ==============================================================================
errorCount = 0
neighbors = [1,3,5,7,9]

iris_data, iris_target = file2matrix('KNN/shuffle_iris.csv')
normMat, ranges, minVals = autoNorm(iris_data)
numTestVecs = int(normMat.shape[0] * 0.25) #0.25 é a % de dados para testar a precisão

#loop para testar varios k vizinhos
for k in range(len(neighbors)):
    for i in range(numTestVecs):
      classifierResult = classify0(normMat[i], normMat[numTestVecs:],iris_target[numTestVecs:], neighbors[k])
      #print("label classificado: %d, label real: %d" % (classifierResult, iris_target[i]))
      if classifierResult != iris_target[i]:
        errorCount += 1
    print('K neighbors:'+str(neighbors[k]))
    # Quanto mais proximo de 0, mais preciso é a predição
    print("error rate: %f" % (errorCount / numTestVecs))
