# This script is used to randomly generate a set of data points, and apply the adaboost method to classify these data points.


# from numpy import *
# import data
# import adaboost

import numpy as np
import math

# Read data from Training or Testing file
def func_readData(filename, option):
    if option == 'train':
        fid = open(filename, 'r')

        label = []
        data = None
        while True:
            fline = fid.readline()
            if len(fline) == 0:  # EOF
                break
            label.append(int(fline[0:fline.find(':')]))

            dataNew = []
            i = fline.find(':') + 1
            dataNew = [float(fline[i:fline.find(',', i, -1)])]
            while True:
                i = fline.find(',', i, -1) + 1
                if not i:
                    break;
                dataNew.append(float(fline[i:fline.find(',', i, -1)]))
            if data is None:
                data = np.mat(dataNew)
            else:
                data = np.vstack([data, np.mat(dataNew)])
        fid.close()
        return data, label
    elif option == 'test':
        fid = open(filename, 'r')
        data = None
        while True:
            fline = fid.readline()
            if len(fline) == 0:  # EOF
                break
            dataNew = []
            i = 0
            while True:
                dataNew.append(float(fline[i:fline.find(',', i, -1)]))
                i = fline.find(',', i, -1) + 1
                if not i:
                    break
            if data is None:
                data = np.mat(dataNew)
            else:
                data = np.vstack([data, np.mat(dataNew)])
        fid.close()
        return data
    else:
        print
        'Wrong input parameter!'


# function for building weak classifiers, i.e.:  stump function

def buildWeakStump(d, l, D):  # (data, label, weight)
    dataMatrix = np.mat(d)
    labelmatrix = np.mat(l).T
    m, n = np.shape(dataMatrix)
    numstep = 10.0
    bestStump = {}
    bestClass = np.mat(np.zeros((5, 1)))
    minErr = np.inf
    for i in range(n):
        datamin = dataMatrix[:, i].min()
        datamax = dataMatrix[:, i].max()
        stepSize = (datamax - datamin) / numstep
        for j in range(-1, int(numstep) + 1):
            for inequal in ['lt', 'gt']:
                threshold = datamin + float(j) * stepSize
                predict = stumpClassify(dataMatrix, i, threshold, inequal)
                err = np.mat(np.ones((m, 1)))
                err[predict == labelmatrix] = 0
                weighted_err = D.T * err;
                if weighted_err < minErr:
                    minErr = weighted_err
                    bestClass = predict.copy()
                    bestStump['dim'] = i
                    bestStump['threshold'] = threshold
                    bestStump['ineq'] = inequal
    return bestStump, minErr, bestClass




# Boosting Algorithm

def train(data, label, numIt=1000):
    eps = 10 ** -16
    weakClassifiers = []
    # m is the number of samples
    m = np.shape(data)[0]
    # sample weights, 1/m at the beginning
    D = np.mat(np.ones((m, 1)) / m)

    estStrong = np.mat(np.zeros((m, 1)))
    for i in range(numIt):
        # bestStump: weak classifier; error: error rate
        bestStump, error, classEstimate = buildWeakStump(data, label, D)

        ##### PLACEHOLDER 1 START ###
        # calculate the weight of the selected decision stump based on its error rate
        alpha = float(.5 * np.log((1.0-error)/max(error, eps)))
        ##### PLACEHOLDER 1 End ###

        # add one more field to bestStump, i.e. classifier weight
        bestStump['alpha'] = alpha
        # add bestStump to the list of weak classifiers
        weakClassifiers.append(bestStump)

        ##### PLACEHOLDER 2 START ###
        # calculate sample weights (of all samples)
        # set sample weights
        # temporary variable to hold the exponent
        temp = np.multiply(-1 * alpha * np.mat(label).T, classEstimate)
        D = np.multiply(D, np.exp(temp))
        # normalize D
        D = D / D.sum()
        ##### PLACEHOLDER 2 End ###

        estStrong += classEstimate * alpha

        EnsembleErrors = np.multiply(np.sign(estStrong) != np.mat(label).T, np.ones((m, 1)))  # Converte to float

        errorRate = EnsembleErrors.sum() / m

        print("current error: {}".format(errorRate))

        if errorRate == 0.0:
            break
    return weakClassifiers

# Use a weak classifier, i.e. a decision stump, to classify data

def stumpClassify(datamat, dim, threshold, inequal):
    res = np.ones((np.shape(datamat)[0], 1))
    if inequal == 'lt':
        res[datamat[:, dim] <= threshold] = -1.0
    else:
        res[datamat[:, dim] > threshold] = -1.0
    return res

# Applying an adaboost classifier for a single data sample

def adaboostClassify(dataTest, classifier):
    dataMatrix = np.mat(dataTest)
    m = np.shape(dataMatrix)[0]
    estStrong = np.mat(np.zeros((m, 1)))
    for i in range(len(classifier)):
        ##### PLACEHOLDER 3 START ###
        # call the function stumpClassify()
        classEstimate = stumpClassify(dataMatrix, m, estStrong, classifier)
        # accumulate all predictions
        estStrong += classifier[i]['alpha'] * classEstimate
        ##### PLACEHOLDER 3 START ###
    return np.sign(estStrong)


# Applying an adaboost classifier for all testing samples
def test(dataSet, classifier):
    label = []
    for i in range(np.shape(dataSet)[0]):
        label.append(adaboostClassify(dataSet[i, :], classifier))
    return label


#############. main ##################
# The data files "train.txt" and "test.txt" are randomly generated by the function randomData() and are used to test your developed codes.

trainData, label = func_readData('train.txt', 'train')
testData = func_readData('test.txt', 'test')

# training
classifier = train(trainData, label, 150)
print('done training\n')
# testing
test(testData, classifier)
print('done testing\n')

