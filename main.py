# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
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
    return openPanda(data)

def zscore(data):
    columns = list(data.columns)
    for col in columns:
        data[col] = (data[col] - data[col].mean())/data[col].std(ddof=0)
    return data

def openPanda(data):
    pandaData = pd.DataFrame(data, dtype=float)
    pandaData = zscore(pandaData)
    return pandaData

#Cost function
def computeCost(X, y, theta):
    inner = np.power(((X * theta.T) - y), 2)
    return np.sum(inner) / (2 * len(X))

#Create a linear regression learner

#Create a logistic regression learner

#Stochastic gradient descent

#Batch gradient descent
def gradientDescent(X, y, theta, alpha, iters):
    temp = np.matrix(np.zeros(theta.shape))
    parameters = int(theta.ravel().shape[1])
    cost = np.zeros(iters)

    # print X
    # print theta.T
    # print y


    for i in range(iters):
        error = (X * theta.T) - y

        for j in range(parameters):
            term = np.multiply(error, X[:,j])
            temp[0,j] = theta[0,j] - ((alpha / len(X)) * np.sum(term))

        theta = temp
        cost[i] = computeCost(X, y, theta)

    return theta, cost

if __name__ == '__main__':
    data = openData("data/spambase.data")

    # append a ones column to the front of the data set
    data.insert(0, 'Ones', 1)

    # set X (training data) and y (target variable)
    cols = data.shape[1]

    # print cols[0]

    X = data.iloc[:,0:cols-1]
    y = data.iloc[:,cols-1:cols]

    print X
    # print X.head()
    # print y.head()

    # convert to matrices and initialize theta
    X = np.matrix(X.values)
    y = np.matrix(y.values)
    theta = np.matrix(np.array(0,0,0,0))

    # print X.head()
    # print y.head()

    # initialize variables for learning rate and iterations
    alpha = 0.01
    iters = 1000

    # perform linear regression on the data set
    g, cost = gradientDescent(X, y, theta, alpha, iters)
    # print g

    # get the cost (error) of the model
    computeCost(X, y, g)




