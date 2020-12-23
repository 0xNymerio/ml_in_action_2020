import pickle
import numpy as np
import re

def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1)
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    if p1 > p0:
        return 1
    else:
        return 0

def createVocabList(dataSet):
    vocabSet = set([])
    for document in dataSet:
        vocabSet = vocabSet | set(document)
    return list(vocabSet)

def trainNB0(trainMatrix, trainCategory):
    numTrainDocs = len(trainMatrix)
    numWords = len(trainMatrix[0])
    pAbusive = sum(trainCategory) / numTrainDocs
    p0Num, p1Num = np.ones(numWords), np.ones(numWords)
    p0Denom, p1Denom = 2.0, 2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect = np.log(p1Num / p1Denom)
    p0Vect = np.log(p0Num / p0Denom)
    return p0Vect, p1Vect, pAbusive

def bagOfWords2VecMN(vocabList, inputSet):
    returnVec = [0] * len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec

def remove_emoji(string):
    emoji_pattern = re.compile("["
                       u"\U0001F600-\U0001F64F"  # emoticons
                       u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                       u"\U0001F680-\U0001F6FF"  # transport & map symbols
                       u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                       u"\U00002702-\U000027B0"
                       u"\U000024C2-\U0001F251"
                       "]+", flags=re.UNICODE)
    return emoji_pattern.sub(r'', string)

def textParse(bigString):
    import re
    listOfTokens = remove_emoji(bigString)
    # Regex no usuario do twitter VirginAmerica e em "your", word mais utilizada.
    listOfTokens = re.sub(r'\bVirginAmerica\b\s+',"",listOfTokens)
    # Retira URL
    listOfTokens = re.sub(r'^https?:\/\/.*[\r\n]*', '', listOfTokens, flags=re.MULTILINE)
    listOfTokens = re.split(r'\W+', listOfTokens)
    return [tok.lower() for tok in listOfTokens if len(tok) > 1]

def testingNB():
    docList, classList = [], []
    # Load and parse
    for i in range(1, 101):
        wordList = textParse(open('./Naive_Bayes/tweets/tweets_negative/%d.txt' % i,
                                  errors='ignore').read())
        docList.append(wordList)
        classList.append(1)
        wordList = textParse(open('./Naive_Bayes/tweets/tweets_positive/%d.txt' % i,
                                  errors='ignore').read())
        docList.append(wordList)
        classList.append(0)
    vocabList = createVocabList(docList)
    # Criar random dataset , utilizando 75 instancias de 200
    trainingSet, testSet = list(range(200)), []
    for i in range(75):
        randIndex = int(np.random.uniform(0, len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat, trainClasses = [], []
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList, docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0V, p1V, pSpam = trainNB0(np.array(trainMat), np.array(trainClasses))
    # Classificar
    errorCount = 0
    for docIndex in testSet:
        wordVector = bagOfWords2VecMN(vocabList, docList[docIndex])
        if classifyNB(np.array(wordVector),
                      p0V, p1V, pSpam) != classList[docIndex]:
            errorCount += 1
            print("classification error", docList[docIndex])
    print('the error rate is: ', errorCount / len(testSet))

testingNB()
