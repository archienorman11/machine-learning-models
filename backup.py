# -*- coding: utf-8 -*-
import sys
from itertools import groupby
import collections

import total as total

data = []
all_lists = []
feature_list = []

def openData(filename):
    print ("loading datset from file %s \n" % filename)
    with open(filename) as infile:
        for line in infile:
            line = line.strip()
            line = line.split(',')
            data.append(line)
    return kFoldData(data)

def kFoldData(data):
    for fold in range(1, 11):
        next = fold
        list = []
        for i, row in enumerate(data):
            if i == next:
                next += 10
                list.append(row) #use i to test the index values being appended
        all_lists.append(list)
    return all_lists

def mu():
    meanFeatures = []
    featureSum = collections.Counter()
    featureCount = collections.Counter()
    total = 4601
    for key, value in feature_list:
        featureSum[key] += float(value)
        featureCount[key] += 1
    for key in sorted(featureSum.iterkeys()):
        meanFeatures.append([key, (featureSum[key]/total)])
    return meanFeatures

def sd(meanFeatures):
    sdFeatures = []
    sumSquares = collections.Counter()
    for id, meanValue in meanFeatures:
        print id
        for key, value in feature_list:
            # print key, value
            if id == key:
                difference = float(value) - meanValue
                differenceSqr = difference * difference
                sumSquares = sum(differenceSqr)
                print sumSquares, id, value

    # for key in sorted(featureSum.iterkeys()):
    #     sdFeatures.append([key, (featureSum[key]/total)])
    return sdFeatures

def preCondition(foldedData):
    count = 0
    for i in range(10):
        for list in foldedData[i]:
            count = count + 1
            for b, featureValue in enumerate(list):
                feature_data = b, featureValue
                feature_list.append(feature_data)


    meanFeatures = mu()
    print(meanFeatures)
    featureSdList = sd(meanFeatures)
    print(featureSdList)
    print(count)


if __name__ == '__main__':
    foldedData = openData("data/spambase.data")
    zScoreData = preCondition(foldedData)






# -*- coding: utf-8 -*-
import numpy as np
import collections
from math import sqrt
from scipy import *
import scipy.stats as ss

np.set_printoptions(threshold=np.nan)

data = []
all_lists = []

def openData(filename):
    print ("loading datset from file %s \n" % filename)
    with open(filename) as infile:
        for line in infile:
            line = line.strip()
            line = line.split(',')
            data.append(line)
    return kFoldData(data)

def kFoldData(data):
    for fold in range(1, 11):
        next = fold
        list = []
        for i, row in enumerate(data):
            if i == next:
                next += 10
                list.append(row) #use i to test the index values being appended
        all_lists.append(list)
    return all_lists

def stats(featureList):
    featureList = featureList.astype(np.float)
    num_items = len(featureList)
    sum_list = sum(featureList)
    mu = sum_list / num_items
    differences = [x - mu for x in featureList]
    squareDiffs = [d ** 2 for d in differences]
    sampleDev = sum(squareDiffs)
    variance =  sampleDev / num_items
    standardDev = sqrt(variance)

    return mu, standardDev


def preCondition(dataset_array):
    dataset_array = dataset_array.astype(np.float)
    list = []
    for i in range(58):
        data = np.array(ss.zscore(dataset_array[:,i]))
        list.append(data)
    list = np.asarray(list)
    return list.T


#Create a linear regression learner

#Create a logistic regression learner

#Stochastic gradient descent

#Batch gradient descent



if __name__ == '__main__':
    foldedData = openData("data/spambase.data")
    np.set_printoptions(precision=3)
    dataset_array = np.array(data)
    zScoreData = preCondition(dataset_array)
    print (zScoreData[6])





