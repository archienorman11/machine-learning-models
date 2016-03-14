# -*- coding: utf-8 -*-
import os
import numpy as np
import pandas as pd
from scipy import *
from pylab import plot, show, xlabel, ylabel
from scipy.special import expit

np.set_printoptions(threshold=np.nan)

data = []


def openData(filename):

    print ("loading datset from file %s \n" % filename)

    with open(filename) as infile:

        for line in infile:

            line = line.strip()

            line = line.split(',')

            data.append(line)

    return openPanda(data)


def openPanda(data):

    pandaData = pd.DataFrame(data, dtype=float)

    pandaData = zscore(pandaData)
    # pandaData = kFoldData(pandaData)

    return pandaData


def zscore(data):

    columns = list(data.columns)

    for col in columns:

        data[col] = (data[col] - data[col].mean()) / data[col].std(ddof=0)

    return data


def prepareData(data):

    data = pd.DataFrame(np.hstack((np.ones((data.shape[0], 1)), data)))

    columns = ['col_' + str(i) for i in range(58)]

    columns.append('y')

    data.columns = columns

    predictions = np.zeros(len(data))

    theta = np.zeros((len(data.columns) - 1, 10))

    return data, columns, predictions, theta


def kFoldData(data):

    for fold in range(1, 11):

        next = fold

        list = []

        for i, row in enumerate(data):

            if i == next:

                next += 10

                list.append(row)  # use i to test the index values being appended

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

    variance = sampleDev / num_items

    standardDev = sqrt(variance)

    return mu, standardDev


def preCondition(dataset_array):

    dataset_array = dataset_array.astype(np.float)

    list = []

    for i in range(58):

        data = np.array(ss.zscore(dataset_array[:, i]))

        list.append(data)

    list = np.asarray(list)

    return list.T

def foldData(fold, data):

    index = range(fold, len(data), 10)

    validation = data.ix[index].values  # validation set

    train = data.ix[~data.index.isin(index)].values

    theta[:, fold]

    return index, train, validation, theta[:, fold]


def linearParams(theta, fold, alpha, X, y, train):

    theta[:, fold] = theta[:, fold] + alpha / (len(train)) * np.dot((y - np.dot(X, theta[:, fold])), X)

    return theta[:, fold]


def logisticParams(theta, fold, alpha, X, y, train):

    theta[:, fold] = theta[:, fold] + alpha / (len(train)) * np.dot((y - expit(np.dot(X, theta[:, fold]))), X)

    return theta[:, fold]

def costFunction(type, theta, fold, validation, index):

    # SSE
    yVal = validation[:, -1]

    XVal = validation[:, :-1]

    # Prediction
    yHat = np.dot(XVal, theta[:, fold])

    predictions[index] = yHat

    if type == 'linear':

        mse = np.sum(np.square(yHat - yVal)) / (2 * len(yVal))

    elif type == 'logistic':

        mse = np.sum(np.square(expit(yHat) - yVal)) / (2 * len(yVal))

    return mse


def gradientDescent(data, theta, alpha, tolerance, type, convergence):

    oldMeanError = 10

    avgMeanError = 5

    iters = 700

    epoch = 0

    avgMseList = []

    while (oldMeanError - avgMeanError) > tolerance and epoch <= iters:

        sqMeansList = []

        for fold in range(10):

            index, train, validation, theta[:, fold] = foldData(fold, data)

            if convergence == 'stocastic':

                for item in range(len(train)):

                    X = train[item, :-1]

                    y = train[item, -1]

                    if type == 'linear':

                        theta[:, fold] = linearParams(theta, fold, alpha, X, y, train)

                    elif type == 'logistic':

                        theta[:, fold] = logisticParams(theta, fold, alpha, X, y, train)

            elif convergence == 'batch':

                X = train[:, :-1]

                y = train[:, -1]

                if type == 'linear':

                    theta[:, fold] = linearParams(theta, fold, alpha, X, y, train)

                elif type == 'logistic':

                    theta[:, fold] = logisticParams(theta, fold, alpha, X, y, train)

            mse = costFunction(type, theta, fold, validation, index)

            sqMeansList.append(mse)

        # Calculate MSE after all folds
        oldMeanError = avgMeanError

        avgMeanError = sum(sqMeansList) / len(sqMeansList)

        avgMseList.append((epoch, avgMeanError))

        print("Ending epoch %d with average mean squared error of %f" % (epoch, avgMeanError))

        epoch += 1

    return avgMseList, epoch


def printResults(result):

    plot(arange(result[0]), result[1])

    xlabel('Iterations')

    ylabel('Cost Function')

    show()


if __name__ == '__main__':

    pandaData = openData("data/spambase.data")

    data, columns, predictions, theta = prepareData(pandaData)

    linStoch1, iters = gradientDescent(data, theta, 0.1, 0.000001, 'linear', 'stocastic')
    printResults(linStoch1)

    # gradientDescent(data, theta, 1, 0.000001, 'linear', 'stocastic')
    # gradientDescent(data, theta, 0.1, 0.000001, 'linear', 'stocastic')
    # gradientDescent(data, theta, 0.2, 0.000001, 'linear', 'stocastic')
    # gradientDescent(data, theta, 0.3, 0.000001, 'linear', 'stocastic')
    # gradientDescent(data, theta, 0.4, 0.000001, 'linear', 'stocastic')

    gradientDescent(data, theta, 1, 0.000001, 'linear', 'batch')
    # gradientDescent(data, theta, 0.1, 0.000001, 'linear', 'batch')
    # gradientDescent(data, theta, 0.2, 0.000001, 'linear', 'batch')
    # gradientDescent(data, theta, 0.3, 0.000001, 'linear', 'batch')
    # gradientDescent(data, theta, 0.4, 0.000001, 'linear', 'batch')
    #
    gradientDescent(data, theta, 1, 0.000001, 'logistic', 'stocastic')
    # gradientDescent(data, theta, 0.1, 0.000001, 'logistic', 'stocastic')
    # gradientDescent(data, theta, 0.2, 0.000001, 'logistic', 'stocastic')
    # gradientDescent(data, theta, 0.3, 0.000001, 'logistic', 'stocastic')
    # gradientDescent(data, theta, 0.4, 0.000001, 'logistic', 'stocastic')
    #
    gradientDescent(data, theta, 1, 0.000001, 'logistic', 'batch')
    # gradientDescent(data, theta, 0.1, 0.000001, 'logistic', 'batch')
    # gradientDescent(data, theta, 0.2, 0.000001, 'logistic', 'batch')
    # gradientDescent(data, theta, 0.3, 0.000001, 'logistic', 'batch')
    # gradientDescent(data, theta, 0.4, 0.000001, 'logistic', 'batch')

    # printResults(linStoch1, iters)
    # print predictions
